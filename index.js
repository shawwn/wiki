var express = require("express");
var app = express();
var childProcess = require('child_process');
var githubUsername = 'shawwn'

app.get("/deploy", function (req, res) { deploy(res); })

app.post("/webhooks/github", function (req, res) {
    /*
    if (!req || !res || !req.body || !req.body.sender || !req.body.ref) { return; }
    var sender = req.body.sender;
    var branch = req.body.ref;

    if(branch.indexOf('master') > -1 && sender.login === githubUsername){
        deploy(res);
    }
    */
    deploy(res);
})

function deploy(res){
    console.log('Deploying...')
    childProcess.exec('./deploy.sh', {maxBuffer: 1024 * 500000}, function(err, stdout, stderr){
        if (err) {
         console.error(err);
         return res.send(500);
        }
	console.log('Deploy successful.')
        res.send(200);
      });
}

server = app.listen(80)
