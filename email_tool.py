import os
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

load_dotenv()


def send_email(to_email, subject, body):

    sender_email = os.getenv("GMAIL_ADDRESS")
    app_password = os.getenv("GMAIL_APP_PASSWORD")

    if not sender_email:
        return "GMAIL_ADDRESS is missing from .env"

    if not app_password:
        return "GMAIL_APP_PASSWORD is missing from .env"

    message = EmailMessage()

    message["From"] = sender_email
    message["To"] = to_email
    message["Subject"] = subject

    message.set_content(body)

    try:

        with smtplib.SMTP("smtp.gmail.com", 587) as server:

            server.starttls()

            server.login(
                sender_email,
                app_password
            )

            server.send_message(message)

        return "Email sent successfully."

    except Exception as e:

        return f"Email failed: {e}"


# -----------------------------
# TEST EMAIL TOOL
# -----------------------------

if __name__ == "__main__":

    recipient = input("Enter recipient email: ")

    result = send_email(
        recipient,
        "KBTCOE AI Agent Test",
        "This is a test email sent by the AI Agent workshop project."
    )

    print("\n" + result)