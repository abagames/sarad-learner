// based on: https://github.com/karpathy/recurrentjs
declare let require: any;
let R = require('../web_modules/recurrentjs/recurrent');

// model parameters
let generator = 'lstm'; // can be 'rnn' or 'lstm'
let hidden_sizes = [100, 100];
let letter_size = 5; // size of letter embeddings

// optimization
let regc = 0.000001; // L2 regularization strength
export let learning_rate = 0.01; // learning rate
let clipval = 5.0; // clip gradients at this value

// prediction params
let sample_softmax_temperature = 1.0; // how peaky model predictions should be
let max_terms_gen = 25; // max length of generated sentences
let max_sentences_gen = 36;

// various global var inits
let epoch_size = -1;
let input_size = -1;
let output_size = -1;
let letterToIndex = {};
let indexToLetter = {};
let vocab = [];
let data_sents = [];
let solver = new R.Solver(); // should be class because it needs memory for step caches

let model = {};

let indentStr = '<IDT>';
let carriageReturnStr = '<CR>';

function initVocab(sents, count_threshold) {
    // go over all characters and keep track of all unique ones seen
    let txt = sents.join(' ').split(' '); // concat all
    // count up all characters
    let d = {};
    for (let i = 0, n = txt.length; i < n; i++) {
        let txti = txt[i];
        if (txti in d) { d[txti] += 1; }
        else { d[txti] = 1; }
    }
    // filter by count threshold and create pointers
    letterToIndex = {};
    indexToLetter = {};
    vocab = [];
    // NOTE: start at one because we will have START and END tokens!
    // that is, START token will be index 0 in model letter vectors
    // and END token will be index 0 in the next character softmax
    let q = 1;
    for (let ch in d) {
        if (d.hasOwnProperty(ch)) {
            if (d[ch] >= count_threshold) {
                // add character to vocab
                letterToIndex[ch] = q;
                indexToLetter[q] = ch;
                vocab.push(ch);
                q++;
            }
        }
    }
    // globals written: indexToLetter, letterToIndex, vocab (list), and:
    input_size = vocab.length + 1;
    output_size = vocab.length + 1;
    epoch_size = sents.length;
}

function utilAddToModel(modelto, modelfrom) {
    for (let k in modelfrom) {
        if (modelfrom.hasOwnProperty(k)) {
            // copy over the pointer but change the key to use the append
            modelto[k] = modelfrom[k];
        }
    }
}

function initModel() {
    // letter embedding vectors
    model = {};
    model['Wil'] = new R.RandMat(input_size, letter_size, 0, 0.08);
    if (generator === 'rnn') {
        let rnn = R.initRNN(letter_size, hidden_sizes, output_size);
        utilAddToModel(model, rnn);
    } else {
        let lstm = R.initLSTM(letter_size, hidden_sizes, output_size);
        utilAddToModel(model, lstm);
    }
}

export function reinit(codeStr: string, modelJson = null) {
    solver = new R.Solver(); // reinit solver
    // process the input, filter out blanks
    let tabReplacedCode = codeStr.replace(/\t/g, indentStr + ' ');
    let data_sents_raw = tabReplacedCode.split('\n');
    data_sents = [];
    let code = '';
    for (let i = 0; i < data_sents_raw.length; i++) {
        let sent = data_sents_raw[i].trim();
        sent = sent.split(/[ ]+/).join(' ');
        if (sent.length > 0) {
            code += `${sent} ${carriageReturnStr} `;
        } else {
            if (code.length > 0) {
                data_sents.push(code.trim());
                code = '';
            }
        }
    }
    initVocab(data_sents, 1); // takes count threshold for characters
    if (modelJson == null) {
        initModel();
    } else {
        loadModel(modelJson);
    }
}

export function saveModel() {
    let out = {};
    out['hidden_sizes'] = hidden_sizes;
    out['generator'] = generator;
    out['letter_size'] = letter_size;
    let model_out = {};
    for (let k in model) {
        if (model.hasOwnProperty(k)) {
            model_out[k] = model[k].toJSON();
        }
    }
    out['model'] = model_out;
    let solver_out = {};
    solver_out['decay_rate'] = solver.decay_rate;
    solver_out['smooth_eps'] = solver.smooth_eps;
    let step_cache_out = {};
    for (let k in solver.step_cache) {
        if (solver.step_cache.hasOwnProperty(k)) {
            step_cache_out[k] = solver.step_cache[k].toJSON();
        }
    }
    solver_out['step_cache'] = step_cache_out;
    out['solver'] = solver_out;
    out['letterToIndex'] = letterToIndex;
    out['indexToLetter'] = indexToLetter;
    out['vocab'] = vocab;
    return JSON.stringify(out);
}

function loadModel(j) {
    hidden_sizes = j.hidden_sizes;
    generator = j.generator;
    letter_size = j.letter_size;
    model = {};
    for (let k in j.model) {
        if (j.model.hasOwnProperty(k)) {
            let matjson = j.model[k];
            model[k] = new R.Mat(1, 1);
            model[k].fromJSON(matjson);
        }
    }
    solver = new R.Solver(); // have to reinit the solver since model changed
    solver.decay_rate = j.solver.decay_rate;
    solver.smooth_eps = j.solver.smooth_eps;
    solver.step_cache = {};
    for (let k in j.solver.step_cache) {
        if (j.solver.step_cache.hasOwnProperty(k)) {
            let matjson = j.solver.step_cache[k];
            solver.step_cache[k] = new R.Mat(1, 1);
            solver.step_cache[k].fromJSON(matjson);
        }
    }
    letterToIndex = j['letterToIndex'];
    indexToLetter = j['indexToLetter'];
    vocab = j['vocab'];
}

function forwardIndex(G, model, ix, prev) {
    let x = G.rowPluck(model['Wil'], ix);
    // forward prop the sequence learner
    let out_struct;
    if (generator === 'rnn') {
        out_struct = R.forwardRNN(G, model, hidden_sizes, x, prev);
    } else {
        out_struct = R.forwardLSTM(G, model, hidden_sizes, x, prev);
    }
    return out_struct;
}

function costfun(model, sent) {
    sent = sent.split(' ');
    // takes a model and a sentence and
    // calculates the loss. Also returns the Graph
    // object which can be used to do backprop
    let n = sent.length;
    let G = new R.Graph();
    let log2ppl = 0.0;
    let cost = 0.0;
    let prev = {};
    for (let i = -1; i < n; i++) {
        // start and end tokens are zeros
        let ix_source = i === -1 ? 0 : letterToIndex[sent[i]]; // first step: start with START token
        let ix_target = i === n - 1 ? 0 : letterToIndex[sent[i + 1]]; // last step: end with END token
        let lh = forwardIndex(G, model, ix_source, prev);
        prev = lh;
        // set gradients into logprobabilities
        let logprobs = lh.o; // interpret output as logprobs
        let probs = R.softmax(logprobs); // compute the softmax probabilities
        log2ppl += -Math.log(probs.w[ix_target]) * Math.LOG2E; // accumulate base 2 log prob and do smoothing
        cost += -Math.log(probs.w[ix_target]);
        // write gradients into log probabilities
        logprobs.dw = probs.w;
        logprobs.dw[ix_target] -= 1
    }
    let ppl = Math.pow(2, log2ppl / (n - 1));
    return { 'G': G, 'ppl': ppl, 'cost': cost };
}

function median(values) {
    values.sort(function(a, b) { return a - b; });
    let half = Math.floor(values.length / 2);
    if (values.length % 2) return values[half];
    else return (values[half - 1] + values[half]) / 2.0;
}

export function learn(count = 100, targetPerplexity = 0) {
    let perplexity = 0;
    for (let i = 0; i < count; i++) {
        perplexity = tick();
        if (i % 10 === 0) {
            console.error
                (`iteration: ${i}  perplexity: ${perplexity.toFixed(2)}`);
        }
        if (perplexity <= targetPerplexity) {
            break;
        }
    }
    return perplexity;
}

function tick() {
    // sample sentence fromd data
    let sentix = R.randi(0, data_sents.length);
    let sent = data_sents[sentix];
    // evaluate cost function on a sentence
    let cost_struct = costfun(model, sent);
    // use built up graph to compute backprop (set .dw fields in mats)
    cost_struct.G.backward();
    // perform param update
    let solver_stats = solver.step(model, learning_rate, regc, clipval);
    return cost_struct.ppl;
}
