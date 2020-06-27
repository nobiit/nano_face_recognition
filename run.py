#!/usr/bin/env python3
# noinspection SpellCheckingInspection


# Class mô tả 1 người
class People(object):
    # Thư mục chứa dữ liệu
    DATA_DIR = 'train_data'

    # Scan những người đã biết
    @staticmethod
    def scan():
        from os import listdir

        return list(map(lambda name: People(name), listdir(People.DATA_DIR)))

    # Tạo ra 1 người giả (Khách)
    @staticmethod
    def register(pid=None):
        from os import mkdir
        from datetime import datetime
        if not pid:
            pid = str(datetime.now().timestamp()).replace('.', '')
        mkdir('%s/guest_%s' % (People.DATA_DIR, pid))
        return People(pid)

    # Khởi tạo
    def __init__(self, pid):
        from os.path import isdir
        from os import mkdir
        self.id = pid
        self._face_encoding = []
        self._last_block = 0
        self._classifier = None
        if not isdir('/'.join([self.DATA_DIR, self.id])):
            mkdir('/'.join([self.DATA_DIR, self.id]))

    # Tạo model nhận diện cho người này
    @property
    def classifier(self):
        rebuild_classifier = False
        if not self._classifier:
            # Nếu số lượng ảnh > 10 mới thực hiện tạo model
            # Do số lượng ảnh thấp -> model sẽ kém chính xác
            if len(self._face_encoding) > 10:
                rebuild_classifier = True
        else:
            # Nếu nhiều hơn 10 ảnh thì cứ 100 ảnh tiếp theo mới tạo lại model
            # Do chi phí (thời gian,bộ nhớ) mỗi lần tạo model là rất lớn
            if len(self._face_encoding) > self._last_block * 100:
                rebuild_classifier = True
        # Nếu thoả mãn 1 trong 2 điều kiện trên thì tạo model
        if rebuild_classifier:
            from sklearn.neighbors import KNeighborsClassifier
            # Sử dụng thuật toán nhận diện gần tương đồng (KNeighbor) với việc ưu tiên khoảng cách
            # Nghĩa là những điểm ở gần (khuôn mặt) sẽ được đánh dấu là giá trị hơn những điểm khác
            classifier = KNeighborsClassifier(algorithm='ball_tree', weights='distance')
            # Load ảnh vào model
            classifier.fit(self._face_encoding, [self.id] * len(self._face_encoding))
            self._classifier = classifier
            self._last_block += 1
        return self._classifier

    # Lấy tên
    @property
    def name(self):
        with open('/'.join([self.DATA_DIR, self.id, 'name.txt']), 'r') as f:
            return f.read().strip()

    # Người này có tin cậy hay không
    @property
    def is_trusted(self):
        with open('/'.join([self.DATA_DIR, self.id, 'trusted.txt']), 'r') as f:
            return f.read().strip().lower() == 'true'

    # Người này có phải khách hay không
    @property
    def is_guest(self):
        return False

    # Tên
    def __str__(self):
        return self.name + (' *' if self.is_trusted else '')

    # Những hình ảnh của người này
    @property
    def images(self):
        from face_recognition.face_recognition_cli import image_files_in_folder
        images = list(filter(
            lambda path: path.startswith('train') and path.endswith('.jpg'),
            image_files_in_folder('/'.join([self.DATA_DIR, self.id]))
        ))
        images.sort()
        return images

    # Cắt khuôn mặt từ các hình ảnh
    @property
    def face_encodes(self):
        def encode_face(path):
            from face_recognition import load_image_file, face_locations, face_encodings
            # Phân tích khuôn mặt
            image = load_image_file(path)
            # Lấy vị trí hình chữ nhật bao quanh khuôn mặt
            boxes = face_locations(image)

            if len(boxes) != 1:
                # Nếu có nhiều hơn 1 khuôn mặt hoặc không tìm thấy khuôn mặt nào cả
                print('[WARNING] Ảnh %s không hợp lệ !' % path)
                return exit(1)
            else:
                # Cắt khuôn mặt được tìm thấy và mã hoá
                return face_encodings(image, known_face_locations=boxes).pop()

        print('Đang phân tích khuôn mặt của %s' % self.name)
        return list(map(encode_face, self.images))

    # Kiểm tra xem người này có ảnh nào không
    @property
    def is_empty(self):
        from os import listdir
        for path in listdir('/'.join([self.DATA_DIR, self.id])):
            if not path.startswith('.'):
                return False
        return True

    # Thêm 1 ảnh
    def add_image(self, image):
        from os.path import isfile
        from cv2 import imwrite
        num = 0
        while isfile('/'.join([self.DATA_DIR, self.id, 'image%s.jpg' % num])):
            num += 1
        imwrite('/'.join([self.DATA_DIR, self.id, 'image%s.jpg' % num]), image)

    # Thêm 1 khuôn mặt đã mã hoá
    def add_encoding(self, faces_encoding):
        self._face_encoding += faces_encoding


# Khách
class Guest(People):
    DATA_DIR = 'guest_data'

    # Khách sẽ không có tên
    @property
    def name(self):
        return 'Guest'

    # Đánh dấu đây là khách
    def is_guest(self):
        return True


# Hướng dẫn đọc dữ liệu
# noinspection SpellCheckingInspection
class Trainer(object):
    # Lấy danh sách người đã biết
    @property
    def peoples(self):
        return People.scan()

    # Tạo model nhận diện chung cho những người đã biết
    def train(self, classifier):
        print('Bắt đầu xây dựng mô hình nhận diện khuôn mặt ...')
        x = []
        y = []
        # Lặp qua từng người
        for people in self.peoples:
            # Lấy dữ liệu khuôn mặt của họ
            face_encodes = people.face_encodes
            x += face_encodes
            y += [people.id] * len(face_encodes)
        # Đưa dữ liệu khuôn mặt của mọi người vào model
        classifier.fit(x, y)
        print('Hoàn thành mô hình nhận diện khuôn mặt !')

    # Tên class
    def __str__(self):
        return 'Trainer'


# Phiên
# Một người được đánh dấu là đã phát hiện trong 1 phiên sẽ không được gửi email cảnh báo lại
class Session(object):
    # Khởi tạo
    def __init__(self, classifier, distance_threshold=0.35):
        self.classifier = classifier
        self.distance_threshold = distance_threshold

    # Tìm kiếm khuôn mặt trong hình
    def find(self, image):
        from face_recognition import face_locations, face_encodings
        # Đảo ngược màu từ RGB qua BGR do OpenCV dùng BGR
        image = image[:, :, ::-1]
        # Lấy toạ độ các hình chữ nhật bao quanh khuôn mặt
        face_locations = face_locations(image)
        # Nếu tìm thấy
        if len(face_locations) > 0:
            # Mã hoá khuôn mặt
            faces_encodings = face_encodings(image, face_locations)
            # So sánh với dữ liệu trong model
            closest_distances = self.classifier.kneighbors(faces_encodings, n_neighbors=1)
            # Tìm dữ liệu có độ chính xác cao nhất
            are_matches = [closest_distances[0][i][0] <= self.distance_threshold for i in range(len(faces_encodings))]

            peoples = []
            for pred, encod, location, rec in zip(self.classifier.predict(faces_encodings), faces_encodings, face_locations, are_matches):
                if rec:
                    # Nếu tìm thấy trong model, liên kết họ với dữ liệu trong model
                    people = People(pred)
                else:
                    # Nếu không tìm thấy thì không liên kết
                    people = None
                peoples.append((people, location, encod))
            return peoples
        return []


# Chương trình chính
# noinspection SpellCheckingInspection
class NanoFaceDetection(object):
    # Thư mục chứa các file test
    TEST_DIR = 'tests'
    # File test
    TEST_FILE = TEST_DIR + '/test.jpg'
    # Model nhận diện
    _classifier = None

    # Khởi tạo
    def __init__(self, admines, scale=4):
        self.admines = admines
        self.scale = scale

    # Sử dụng model nhận diện hiện tại hoặc tạo mới
    @property
    def classifier(self):
        if not self._classifier:
            try:
                # Đọc model được lưu trữ trên đĩa nếu có
                with open('%s/classifier.dat' % People.DATA_DIR, 'rb') as f:
                    from pickle import Unpickler
                    self._classifier = Unpickler(f).load()
            except FileNotFoundError:
                # Nếu không có model cũ
                from sklearn.neighbors import KNeighborsClassifier
                # Khởi tạo model sử dụng thuật toán KNeighbor với ưu tiên khoảng cách
                self._classifier = KNeighborsClassifier(algorithm='ball_tree', weights='distance')
                # Khởi tạo hàm đào tạo model
                trainer = Trainer()
                # Đào tạo model
                trainer.train(self._classifier)
                # Lưu lại model lên đĩa
                with open('%s/classifier.dat' % People.DATA_DIR, 'wb') as f:
                    from pickle import Pickler
                    Pickler(f).dump(self._classifier)
        return self._classifier

    # Xoá model trên đĩa
    @staticmethod
    def clean_classifier():
        from os import remove
        remove('%s/classifier.dat' % People.DATA_DIR)

    # Chạy
    def run(self, test=False, webcam=False, only=False):
        print('Khởi động Nano FaceDetection !')
        try:
            if test:
                self.processing_test(self.classifier, only=only)
            if webcam:
                self.processing_webcam(self.classifier)
        except KeyboardInterrupt:
            pass

    # Vẽ các ô vuông bao quanh khuôn mặt
    @staticmethod
    def draw_info(frame, position, name=None, is_warning=False, guest_id=None, time=None, count=None, alerted=False):
        from cv2 import rectangle, putText, FONT_HERSHEY_DUPLEX, FILLED
        (top, right, bottom, left) = position
        title = name
        if not title:
            title = ''
        # Nếu là khách thì hiện mã nhận diện của khách
        if guest_id is not None:
            title += ' #%s' % guest_id
        # Thời gian kể từ lúc được tìm thấy
        if count is not None and time is not None:
            title += ' %s - %ss' % (count, time)
        if name and not name.startswith('Guest'):
            # Nếu là người quen
            # Vẽ ô vuông màu xanh lá cây
            rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255) if is_warning else (0, 255, 0), FILLED)
            # Vẽ tên
            putText(frame, title, (left + 6, bottom - 6), FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
        else:
            # Nếu là khách
            # Vẽ màu đỏ nếu chưa nhận diện, ngược lại vẽ màu xanh nước biển nếu đã nhận diện
            rectangle(frame, (left, top), (right, bottom), (255, 0, 0) if alerted else (0, 0, 255), 5)
            rectangle(frame, (left, bottom - 35), (right, bottom), (255, 0, 0) if alerted else (0, 0, 255), FILLED)
            # Vẽ chữ
            putText(frame, title, (left + 6, bottom - 6), FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    # Chạy các file trong thư mục test
    def processing_test(self, classifier, only=False):
        from os import environ
        from face_recognition.face_recognition_cli import image_files_in_folder
        from cv2 import imread, resize, imshow, waitKey
        try:
            if only:
                images = [self.TEST_FILE]
            else:
                images = list(filter(
                    lambda image_path: image_path.startswith('test') and image_path.endswith('.jpg'),
                    image_files_in_folder(self.TEST_DIR)
                ))
        except FileNotFoundError:
            return
        images.sort()
        # Khởi tạo phiên
        session = Session(classifier)
        # Lặp qua từng ảnh
        for path in images:
            print('Tìm thấy ảnh %s - Đang thực hiện phân tích ...' % path)
            # Đọc ảnh
            image = imread(path)
            # Thu nhỏ ảnh đi scale lần
            _image = resize(image, (0, 0), fx=1 / self.scale, fy=1 / self.scale)
            # Tìm kiếm khuôn mặt trong ảnh
            peoples = session.find(_image)
            if len(peoples) > 0:
                # Lặp qua từng kết quả
                for people, location, face_encoding in peoples:
                    top, right, bottom, left = location

                    # Kích thước thật của khuôn mặt
                    top *= self.scale
                    right *= self.scale
                    bottom *= self.scale
                    left *= self.scale

                    if people:
                        print('Phát hiện %s trong ảnh %s' % (people.name, path))
                        # Vẽ thông tin người quen lên ảnh
                        self.draw_info(image, (top, right, bottom, left), name=people.name)
                    else:
                        # Vẽ thông tin khách lên ảnh
                        self.draw_info(image, (top, right, bottom, left), name='Guest')
                        print('Phát hiện ra người lạ trong ảnh %s' % path)
            else:
                print('Không tìm thấy ai trong ảnh %s' % path)

            # Nếu có thể hiển thị
            if environ.get('DISPLAY'):
                _image = image
                # Thu nhỏ ảnh để vừa với màn hình
                while _image.shape[0] > 1000 or _image.shape[1] > 1000:
                    _image = resize(_image, (0, 0), fx=0.9, fy=0.9)
                # Hiện ảnh
                imshow('Image ' + path, _image)
                waitKey(1)
        # Giữ chương trình không tắt
        while True:
            if waitKey(1) & 0xFF == ord('q'):
                break

    # Gửi mail
    def send_mail(self, metadata):
        from datetime import datetime
        from mail import SendAlertStranger
        # Đánh dấu thời điểm gửi mail
        metadata['last_alert'] = datetime.now()
        # Gửi mail
        return SendAlertStranger(self.admines, metadata).start()

    # Chạy từ webcam
    def processing_webcam(self, classifier):
        from os import environ
        from cv2 import VideoCapture, resize, imshow, waitKey
        # from bdc import load_known_faces
        # Mở camera
        video_capture = VideoCapture(0)
        # Mở phiên
        session = Session(classifier)
        # Đánh dấu trạng thái
        is_empting = False
        # load_known_faces()
        # Nếu camera còn mở
        while video_capture.isOpened():
            # Đọc ảnh từ camera
            ret, frame = video_capture.read()
            # Giảm ảnh đi scale lần
            _frame = resize(frame, (0, 0), fx=1 / self.scale, fy=1 / self.scale)
            # Đảo ngược màu
            _frame = _frame[:, :, ::-1]
            # Tìm kiếm khuôn mặt trong ảnh
            peoples = session.find(_frame)
            if len(peoples) > 0:
                # Lặp qua từng khuôn mặt
                for people, location, face_encoding in peoples:
                    top, right, bottom, left = location

                    # Tính toán lại kích thước ban đầu
                    top *= self.scale
                    right *= self.scale
                    bottom *= self.scale
                    left *= self.scale

                    # Cắt khuôn mặt khỏi hình
                    face_image = frame[top:bottom, left:right]
                    # Đưa khuôn mặt về 500x500
                    face_image = resize(face_image, (500, 500))

                    # Nếu tìm ra thông tin họ
                    if people:
                        print('Phát hiện %s%s' % (people.name, '( WARNING )' if not people.is_trusted else ''))
                        # Vẽ thông tin người này lên ảnh và cảnh báo nếu người này không được đánh dấu là tin cậy
                        self.draw_info(frame, (top, right, bottom, left), people.name, is_warning=not people.is_trusted)
                    else:
                        # Nếu không tìm ra thông tin họ -> họ là khách
                        from datetime import datetime
                        from face_recognition import face_locations, face_encodings
                        from bdc import lookup_known_face, register_new_face
                        # Lấy dữ liệu về họ
                        metadata = lookup_known_face(face_encoding)

                        # Nếu có thông tin về họ -> họ đã ghé thăm trước đây
                        if metadata is not None:
                            # Tính toán thời gian họ có mặt kể từ lúc đầu tiên thấy họ
                            time_at_door = datetime.now() - metadata['first_seen_this_interaction']
                            time = int(time_at_door.total_seconds())
                        else:
                            # Nếu không có thông tin -> khách mới
                            metadata = register_new_face(face_encoding, ('Camera', datetime.now(), face_image))
                            time = 0

                        # Nếu họ có ít hơn 10 ảnh thì chụp họ, nếu nhiều hơn thì cứ sau 10s thì chụp thêm 1 ảnh về họ
                        if time % 10:
                            metadata['face_image'] = metadata['face_image'][1:10]
                        if len(metadata['face_image']) < 10:
                            print('Cập nhật ảnh cho %s' % metadata['id'])
                            metadata['face_image'].append(('Camera', datetime.now(), face_image))

                        # Kiểm tra thông báo
                        alerted = False
                        # Nếu lần cuối thông báo trong 60s trước thì tô màu họ bằng màu xanh (để đánh dấu là đã gửi email)
                        if metadata.get('last_alert'):
                            alerted = (datetime.now() - metadata['last_alert']).total_seconds() < 60
                        else:
                            # Nếu chưa thông báo và họ có mặt hơn 10s
                            if time > 10:
                                # Thì gửi mail
                                self.send_mail(metadata)
                                alerted = True

                        print('Phát hiện ra người lạ (id=%s, count=%s, time=%s, alerted=%s)' %
                              (metadata['id'], metadata['seen_count'], time, 'true' if alerted else 'false'))
                        # Vẽ thông tin họ
                        self.draw_info(frame, (top, right, bottom, left), alerted=alerted, guest_id=metadata['id'], count=metadata['seen_count'],
                                       time=time)
                    # Đánh dấu là có người
                    is_empting = False
            else:
                # Thông báo không tìm thấy ai
                if not is_empting:
                    print('Không tìm thấy ai trong ảnh')
                # Đánh dấu là không tìm thấy để tránh thông báo lặp lại
                is_empting = True
            # Nếu hỗ trợ hiển thị
            if environ.get('DISPLAY'):
                imshow('Video', frame)
                waitKey(1)
            if waitKey(1) & 0xFF == ord('q'):
                break

    # Tên class
    def __str__(self):
        return 'NanoFaceDetection'


# Nếu file được chạy trực tiếp
# noinspection SpellCheckingInspection
if __name__ == '__main__':
    from sys import argv

    # Khởi tạo FaceDetection và thiết lập danh sách người nhận mail
    nano = NanoFaceDetection(admines=[
        ('Chiến', 'nguyenchienbg2k@gmail.com'),
    ])
    # if len(argv) == 1:
    #     exit(nano.run(test=True, webcam=True))
    # Nếu có tham số
    if len(argv) == 2:
        # Lấy tham số
        command = argv.copy().pop()
        # Nếu là reset
        if command == 'reset':
            exit(nano.clean_classifier())
        # Nếu là test
        if command == 'test':
            exit(nano.run(test=True))
        # Nếu là webcam
        # noinspection SpellCheckingInspection
        if command == 'webcam':
            exit(nano.run(webcam=True))
        # Nếu là 1 link web
        if command.startswith('http://') or command.startswith('https://'):
            from requests import get

            # Tải file
            res = get(command)
            # Mở file test
            with open(self.TEST_FILE, 'wb') as f:
                # Lưu file vào file test
                f.write(res.content)
                # Đóng file
                f.close()

            # Chạy test
            exit(nano.run(test=True, only=True))
    # Cảnh báo nếu lệnh không hợp lệ
    # noinspection SpellCheckingInspection
    print('Lệnh không hợp lệ !')
    # noinspection SpellCheckingInspection
    print('Cấu trúc: run-face-recognition [reset|test|webcam|<url>]')
