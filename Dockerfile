FROM truongthoigian/php_vld:latest
ADD test.py /src/test.py
ENTRYPOINT ["/usr/bin/python","/src/test.py"]
# Start the service
# CMD ["/usr/sbin/httpd", "-D", "FOREGROUND"]
