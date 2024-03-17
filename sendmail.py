import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os
import streamlit as st

# Default configuration
sender_email = '20jr1a05b5@gmail.com'
receiver_email = '20jr1a05b5@gmail.com'
password = 'yubppfimevmxjceg'
subject = 'Unknown person is Identified'
# body = 'Please find the attached video.'
html_body = """
<html>
    <body>
        <h1 style="color:red">ALERT!</h1>
        <h2>Unknown person is recognized at your home...</h2>
        <h2>Have a look Boss!</h2>
    </body>
</html>
"""

def send_email(attachment_path, sender=sender_email, receiver=receiver_email, pwd=password, subject=subject, html=html_body):
    try:
        # Create a multipart message
        msg = MIMEMultipart()
        msg['From'] = sender
        msg['To'] = receiver
        msg['Subject'] = subject

        # Attach the HTML body of the email
        msg.attach(MIMEText(html, 'html'))
        # Attach the body of the email
        # msg.attach(MIMEText(body, 'plain'))
        # Attach the video file
        with open(attachment_path, 'rb') as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', rf'attachment; filename={attachment_path}')
            msg.attach(part)

        # Connect to the SMTP server and send the email
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(sender, pwd)
        server.send_message(msg)
        server.quit()

        return True  # Email sent successfully
    except Exception as e:
        print(f"Error sending email: {e}")
        st.write(f"Error sending email: {e}")
        return False  # Email sending failed
