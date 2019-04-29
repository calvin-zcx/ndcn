from twilio.rest import Client


class SMS:
    def __init__(self):
        self.account_sid = r'AC767d0873e45bea0de402eb2c39a9a19e'
        self.auth_token = r'3f9ac909ef56115e9ec85f1fcbe37d61'
        self.client = Client(self.account_sid, self.auth_token)

    def send_sms(self, body):
        message = self.client.messages.create(
            body=body,
            from_='+14798888163',
            to='+12129200603'
        )
        return message
