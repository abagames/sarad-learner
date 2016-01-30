import * as fs from 'fs';
import * as program from 'commander';
import * as learner from './learner';
declare let require: any;
let process = require('process');

program
    .version('1.0.0')
    .option('-i --iterCnt <n>', 
    'iteration count (default: 100)', '100')
    .option('-m --model <file_name>', 
    'initial model file name (default: null)', null)
    .option('-l --learnRate <n>', 
    'learning rate (default: 0.01)', '0.01')
    .parse(process.argv);
let args = <any>program;
let iterationCount = Number(args.iterCnt);
let modelJson = null;
if (args.model != null) {
    modelJson = JSON.parse(fs.readFileSync(args.model, {encoding: 'utf8'}));
}
learner.learning_rate = Number(args.learnRate);

let stdinStr = "";
process.stdin.resume();
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => {
    stdinStr += chunk;
});
process.stdin.on('end', () => {
    learner.reinit(stdinStr, modelJson);
    learner.learn(iterationCount);
    console.log(learner.saveModel());
});
