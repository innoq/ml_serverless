This example runs currently in offline mode only, with https://serverless.com/ framework + 'serverless-offline' plugin. The reason why deployment on AWS fails is a limit to 250 MBs considering deployment artifact unpackaged. The suggested way how get this work is to use 'minify' and 'uglify' approaches for JS files. Feel free to contribute!

