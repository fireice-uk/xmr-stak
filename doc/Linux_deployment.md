# Deploying portable **XMR-Stak** on Linux systems

XMR-Stak releases include a pre-built portable version. If you are simply using it to avoid having to compile the application, you can simply download **xmr-stak-portbin-linux.tar.gz** from our [latest releases](https://github.com/fireice-uk/xmr-stak/releases/latest). Open up command line, and use the following commands:

```
tar xzf xmr-stak-portbin-linux.tar.gz
./xmr-stak.sh
```

Configuration and tuning files will be generated automatically from your answers.

For automatic deployments, please use the steps above to obtain config.txt and use the following script:

```
#!/bin/bash
curl -O `curl -s https://api.github.com/repos/fireice-uk/xmr-stak/releases/latest | grep -o 'browser_download_url.*xmr-stak-portbin-linux.tar.gz' | sed 's/.*"//'`
curl -O http://path.to/your/config.txt
tar xzf xmr-stak-portbin-linux.tar.gz
./xmr-stak.sh
```

XMR-Stak will auto-configure and go to work. You don't even need Docker!


