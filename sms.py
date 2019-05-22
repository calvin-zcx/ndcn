from twilio.rest import Client


class SMS:
    def __init__(self):
        self.account_sid = r'DELETED'
        self.auth_token = r'DELETED'
        self.client = Client(self.account_sid, self.auth_token)

    def send_sms(self, body):
        message = self.client.messages.create(
            body=body,
            from_='DELETED',
            to='DELETED'
        )
        return message
