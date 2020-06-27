#!/usr/bin/env python3
from threading import Thread


# Lỗi chưa cấu hình gửi mail
class NotConfigured(Exception):
    pass


# Lỗi không đúng quy trình gửi mail
class FlowException(Exception):
    pass


# Class mô tả 1 email
class Message(object):
    def __init__(self):
        super().__init__()
        self._id = None
        self.mail = None

    # ID của email
    @property
    def id(self):
        from email.utils import make_msgid

        if not self._id:
            self._id = make_msgid(domain='alert.nano.local')
        return self._id

    # Tiêu đề
    @property
    def subject(self):
        raise FlowException

    # Nội dung
    @property
    def content(self):
        raise FlowException

    # Đính kèm
    @property
    def attachments(self):
        return []

    # Người gửi
    @property
    def from_address_encoded(self):
        from email.header import Header
        if self.mail:
            name, address = self.mail.me
            return '%s <%s>' % (Header(name).encode(), address)
        raise FlowException

    # Người nhận
    @property
    def to_address_encoded(self):
        from email.header import Header
        if self.mail:
            name, address = self.mail.master
            return '%s <%s>' % (Header(name).encode(), address)
        raise FlowException

    # Người CC
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

    # Tiêu đề
    @property
    def subject_encoded(self):
        from email.header import Header
        return Header('[NANO] %s' % self.subject).encode()

    # Dữ liệu meta của Email
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

    # Dữ liệu chính của Email
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


# Class mô tả 1 kết nối tới máy chủ
class Mail(object):
    def __init__(self):
        super().__init__()
        self._imap = None
        self._smtp = None
        self.me = (None, None)
        self.address = []
        self.master = (None, None)

    # Kết nối Imap để đọc thư
    @property
    def imap(self):
        if not self._imap:
            raise NotConfigured
        return self._imap

    # Kết nối SMTP để gửi thư
    @property
    def smtp(self):
        if not self._smtp:
            raise NotConfigured
        return self._smtp

    # Đăng nhập
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

    # Thiết lập người gửi
    def set_from(self, name, email):
        self.me = (name, email)

    # Thiết lập người nhận
    def set_to(self, address):
        self.address = address

    # Đăng nhập Gmail
    def login_gmail(self, username, password):
        return self.login('imap.gmail.com', 'smtp.gmail.com', username, password)

    # Email người gửi
    @property
    def from_address(self):
        name, address = self.me
        return address

    # Email người nhận
    @property
    def to_address(self):
        name, address = self.master
        return address

    # Gửi tới 1 địa chỉ
    def send_to(self, message, address):
        self.master = address
        print('Đang gửi tới %s (%s)' % self.master)
        self.smtp.sendmail(
            self.from_address,
            self.to_address,
            message.data.as_string(),
        )

    # Gửi thư
    def send(self, message):
        if self.address:
            message.mail = self
            # Lặp qua từng người nhận
            for address in self.address:
                # Và gửi
                self.send_to(message, address)


# Email thông báo người lạ
class AlertStranger(Message):
    # Khởi tạo
    def __init__(self, metadata):
        super().__init__()
        self.metadata = metadata
        self._attachments = []

    # Tiêu đề
    @property
    def subject(self):
        return 'Bạn có một người lạ ghé thăm !'

    # Trích xuất 1 ảnh và đính kèm
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

    # Xuất ảnh và đính kèm
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

    # Đính kèm
    @property
    def attachments(self):
        return self._attachments

    # Tạo mã HTML cho 1 ảnh
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

    # Tạo mã HTML cho tất cả ảnh
    @property
    def html_images(self):
        def build_image(image):
            camera, time, raw = image
            return self.build_html_image(camera, time, raw)

        left, right = self.images
        return ''.join(list(map(build_image, left))), ''.join(list(map(build_image, right)))

    # Nội dung Email
    @property
    def content(self):
        with open('./alert-stranger.html', 'r') as f:
            html = f.read()
        left, right = self.html_images
        html = html.replace('{{left}}', left)
        html = html.replace('{{right}}', right)
        return html


# Gửi email cảnh báo người lạ
class SendAlertStranger(Thread):
    # Khởi tạo
    def __init__(self, address, metadata):
        self.address = address
        self.metadata = metadata
        super().__init__()

    # Gửi email
    def run(self):
        if self.metadata:
            print('Chuẩn bị gửi mail !')
            # Khởi tạo
            mail = Mail()
            # Thiết lập người gửi
            mail.set_from('Nano Alert System', 'dyq1302@gmail.com')
            # Thiết lập người nhận
            mail.set_to(self.address)
            # Đăng nhập Gmail
            mail.login_gmail('dyq1302@gmail.com', 'DyQ2208130298')
            # Gửi mail
            mail.send(AlertStranger(self.metadata))
            print('Đã gửi mail !')
