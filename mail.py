#!/usr/bin/env python3
from threading import Thread


class NotConfigured(Exception):
    pass


class FlowException(Exception):
    pass


class Message(object):
    def __init__(self):
        super().__init__()
        self._id = None
        self.mail = None

    @property
    def id(self):
        from email.utils import make_msgid

        if not self._id:
            self._id = make_msgid(domain='alert.nano.local')
        return self._id

    @property
    def subject(self):
        raise FlowException

    @property
    def content(self):
        raise FlowException

    @property
    def attachments(self):
        return []

    @property
    def from_address_encoded(self):
        from email.header import Header
        if self.mail:
            name, address = self.mail.me
            return '%s <%s>' % (Header(name).encode(), address)
        raise FlowException

    @property
    def to_address_encoded(self):
        from email.header import Header
        if self.mail:
            name, address = self.mail.master
            return '%s <%s>' % (Header(name).encode(), address)
        raise FlowException

    @property
    def cc_address_encoded(self):
        from email.header import Header
        if self.mail:
            raw = []
            for mail_address in self.mail.address:
                if mail_address == self.mail.master:
                    continue
                name, address = mail_address
                raw.append('%s <%s>' % (Header(name).encode(), address))
            return ','.join(raw)
        raise FlowException

    @property
    def subject_encoded(self):
        from email.header import Header
        return Header('[NANO] %s' % self.subject).encode()

    @property
    def header(self):
        from email.utils import formatdate
        headers = {
            # 'MIME-Version': '1.0',
            'Date': formatdate(),
            'Message-ID': self.id,
            'Subject': self.subject_encoded,
            'From': self.from_address_encoded,
            'To': self.to_address_encoded,
            'Reply-To': self.to_address_encoded,
        }

        if self.cc_address_encoded:
            headers['Cc'] = self.cc_address_encoded

        return headers

    @property
    def data(self):
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        msg = MIMEMultipart()
        for k, v in self.header.items():
            msg[k] = v
        msg.attach(MIMEText(self.content, 'html'))
        for attachment in self.attachments:
            msg.attach(attachment)
        return msg


class Mail(object):
    def __init__(self):
        super().__init__()
        self._imap = None
        self._smtp = None
        self.me = (None, None)
        self.address = []
        self.master = (None, None)

    @property
    def imap(self):
        if not self._imap:
            raise NotConfigured
        return self._imap

    @property
    def smtp(self):
        if not self._smtp:
            raise NotConfigured
        return self._smtp

    def login(self, imap, smtp, username, password):
        from imaplib import IMAP4_SSL
        from smtplib import SMTP_SSL
        if imap:
            self._imap = IMAP4_SSL(imap)
            self.imap.noop()
            self.imap.login(username, password)
            # self.imap.ping()
        if smtp:
            self._smtp = SMTP_SSL(smtp)
            self.smtp.noop()
            self.smtp.login(username, password)
            # self.smtp.ping()

    def set_from(self, name, email):
        self.me = (name, email)

    def set_to(self, address):
        self.address = address

    def login_gmail(self, username, password):
        return self.login('imap.gmail.com', 'smtp.gmail.com', username, password)

    @property
    def from_address(self):
        name, address = self.me
        return address

    @property
    def to_address(self):
        name, address = self.master
        return address

    def send_to(self, message, address):
        self.master = address
        print('Đang gửi tới %s (%s)' % self.master)
        self.smtp.sendmail(
            self.from_address,
            self.to_address,
            message.data.as_string(),
        )

    def send(self, message):
        if self.address:
            message.mail = self
            for address in self.address:
                self.send_to(message, address)


class AlertStranger(Message):
    def __init__(self, metadata):
        super().__init__()
        self.metadata = metadata
        self._attachments = []

    @property
    def subject(self):
        return 'Bạn có một người lạ ghé thăm !'

    @staticmethod
    def extract_image(image):
        from cv2 import imwrite
        from random import randint
        from os import remove
        path = '/tmp/' + str(randint(1000, 99999)) + '.jpg'
        imwrite(path, image)
        content = open(path, 'rb').read()
        remove(path)
        return content

    @property
    def images(self):
        left = []
        right = []
        index = 0
        for camera, time, image in self.metadata['face_image']:
            index += 1
            print('Xử lý ảnh %s' % index)
            camera = camera + ' ' + str(index)
            time = time.strftime('%H:%M %d/%m/%Y')
            image = AlertStranger.extract_image(image)
            data = (camera, time, image)
            if index % 2 == 1:
                left.append(data)
            else:
                right.append(data)
        return left, right

    @property
    def attachments(self):
        return self._attachments

    def build_html_image(self, camera, time, raw):
        from email.mime.base import MIMEBase
        from email.encoders import encode_base64
        image = MIMEBase('image', 'jpeg')
        image['Content-ID'] = '<%s>' % ('image_%s' % len(self._attachments))
        image.set_payload(raw)
        encode_base64(image)
        image.add_header(
            'Content-Disposition',
            f'attachment; filename=%s' % ('image%s.jpg' % len(self._attachments)),
        )
        self._attachments.append(image)
        return '\n'.join([
            '<tr>',
            '    <td style="padding-top: 20px; padding-right: 10px;">',
            '       <a href="#">',
            '           <img '
            '               src="cid:image_%s"' % str(len(self._attachments) - 1),
            '               alt=""',
            '               style="width: 100%; max-width: 600px; height: auto; margin: auto; display: block;"',
            '           >',
            '       </a>',
            '       <div class="text-project" style="text-align: center;">',
            '           <h3>',
            '               <a href="#">%s</a>' % camera,
            '           </h3>',
            '           <span>%s</span>' % time,
            '        </div>',
            '    </td>',
            '</tr>'
        ])

    @property
    def html_images(self):
        def build_image(image):
            camera, time, raw = image
            return self.build_html_image(camera, time, raw)

        left, right = self.images
        return ''.join(list(map(build_image, left))), ''.join(list(map(build_image, right)))

    @property
    def content(self):
        with open('./alert-stranger.html', 'r') as f:
            html = f.read()
        left, right = self.html_images
        html = html.replace('{{left}}', left)
        html = html.replace('{{right}}', right)
        return html


class SendAlertStranger(Thread):
    def __init__(self, address, metadata):
        self.address = address
        self.metadata = metadata
        super().__init__()

    def run(self):
        if self.metadata:
            print('Chuẩn bị gửi mail !')
            mail = Mail()
            mail.set_from('Nano Alert System', 'dyq1302@gmail.com')
            mail.set_to(self.address)
            mail.login_gmail('dyq1302@gmail.com', 'DyQ2208130298')
            mail.send(AlertStranger(self.metadata))
            print('Đã gửi mail !')
