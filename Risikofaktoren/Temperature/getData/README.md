Get the data from the Merida SFTP server:
https://merida.rse-web.it/?language=EN#download1

There is the password for the SFTP in the file

Create a .env file with the following content:
SFTP_USERNAME=merida
SFTP_PASSWORD=<your_password>

Tipp, the password in in the file in the format:
--user "merida:<Password>"
