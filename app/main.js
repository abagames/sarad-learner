var fs = require('fs');
var program = require('commander');
var learner = require('./learner');
var process = require('process');
program
    .version('1.0.0')
    .option('-i --iterCnt <n>', 'iteration count (default: 100)', '100')
    .option('-m --model <file_name>', 'initial model file name (default: null)', null)
    .option('-l --learnRate <n>', 'learning rate (default: 0.01)', '0.01')
    .parse(process.argv);
var args = program;
var iterationCount = Number(args.iterCnt);
var modelJson = null;
if (args.model != null) {
    modelJson = JSON.parse(fs.readFileSync(args.model, { encoding: 'utf8' }));
}
learner.learning_rate = Number(args.learnRate);
var stdinStr = "";
process.stdin.resume();
process.stdin.setEncoding('utf8');
process.stdin.on('data', function (chunk) {
    stdinStr += chunk;
});
process.stdin.on('end', function () {
    learner.reinit(stdinStr, modelJson);
    learner.learn(iterationCount);
    console.log(learner.saveModel());
});
//# sourceMappingURL=main.js.map