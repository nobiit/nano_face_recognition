#!/usr/bin/env python3


# noinspection SpellCheckingInspection


class People(object):
    DATA_DIR = 'train_data'

    @staticmethod
    def scan():
        from os import listdir

        return list(map(lambda name: People(name), listdir(People.DATA_DIR)))

    @staticmethod
    def register(pid=None):
        from os import mkdir
        from datetime import datetime
        if not pid:
            pid = str(datetime.now().timestamp()).replace('.', '')
        mkdir('%s/guest_%s' % (People.DATA_DIR, pid))
        return People(pid)

    def __init__(self, pid):
        from os.path import isdir
        from os import mkdir
        self.id = pid
        self._face_encoding = []
        self._last_block = 0
        self._classifier = None
        if not isdir('/'.join([self.DATA_DIR, self.id])):
            mkdir('/'.join([self.DATA_DIR, self.id]))

    @property
    def classifier(self):
        rebuild_classifier = False
        if not self._classifier:
            if len(self._face_encoding) > 10:
                rebuild_classifier = True
        else:
            if len(self._face_encoding) > self._last_block * 100:
                rebuild_classifier = True
        if rebuild_classifier:
            from sklearn.neighbors import KNeighborsClassifier
            classifier = KNeighborsClassifier(algorithm='ball_tree', weights='distance')
            classifier.fit(self._face_encoding, [self.id] * len(self._face_encoding))
            self._classifier = classifier
            self._last_block += 1
        return self._classifier

    @property
    def name(self):
        with open('/'.join([self.DATA_DIR, self.id, 'name.txt']), 'r') as f:
            return f.read().strip()

    @property
    def is_trusted(self):
        with open('/'.join([self.DATA_DIR, self.id, 'trusted.txt']), 'r') as f:
            return f.read().strip().lower() == 'true'

    @property
    def is_guest(self):
        return False

    def __str__(self):
        return self.name + (' *' if self.is_trusted else '')

    @property
    def images(self):
        from face_recognition.face_recognition_cli import image_files_in_folder
        images = list(filter(
            lambda path: path.startswith('train') and path.endswith('.jpg'),
            image_files_in_folder('/'.join([self.DATA_DIR, self.id]))
        ))
        images.sort()
        return images

    @property
    def face_encodes(self):
        def encode_face(path):
            from face_recognition import load_image_file, face_locations, face_encodings
            image = load_image_file(path)
            boxes = face_locations(image)

            if len(boxes) != 1:
                print('[WARNING] Ảnh %s không hợp lệ !' % path)
                return exit(1)
            else:
                return face_encodings(image, known_face_locations=boxes).pop()

        print('Đang phân tích khuôn mặt của %s' % self.name)
        return list(map(encode_face, self.images))

    @property
    def is_empty(self):
        from os import listdir
        for path in listdir('/'.join([self.DATA_DIR, self.id])):
            if not path.startswith('.'):
                return False
        return True

    def add_image(self, image):
        from os.path import isfile
        from cv2 import imwrite
        num = 0
        while isfile('/'.join([self.DATA_DIR, self.id, 'image%s.jpg' % num])):
            num += 1
        imwrite('/'.join([self.DATA_DIR, self.id, 'image%s.jpg' % num]), image)

    def add_encoding(self, faces_encoding):
        self._face_encoding += faces_encoding


class Guest(People):
    DATA_DIR = 'guest_data'

    @property
    def name(self):
        return 'Guest'

    def is_guest(self):
        return True


# noinspection SpellCheckingInspection
class Trainer(object):
    @property
    def peoples(self):
        return People.scan()

    def train(self, classifier):
        print('Bắt đầu xây dựng mô hình nhận diện khuôn mặt ...')
        x = []
        y = []
        for people in self.peoples:
            face_encodes = people.face_encodes
            x += face_encodes
            y += [people.id] * len(face_encodes)
        classifier.fit(x, y)
        print('Hoàn thành mô hình nhận diện khuôn mặt !')

    def __str__(self):
        return 'Trainer'


class Session(object):
    def __init__(self, classifier, distance_threshold=0.35):
        self.classifier = classifier
        self.distance_threshold = distance_threshold

    def find(self, image):
        from face_recognition import face_locations, face_encodings
        image = image[:, :, ::-1]
        face_locations = face_locations(image)
        if len(face_locations) > 0:
            faces_encodings = face_encodings(image, face_locations)
            closest_distances = self.classifier.kneighbors(faces_encodings, n_neighbors=1)
            are_matches = [closest_distances[0][i][0] <= self.distance_threshold for i in range(len(faces_encodings))]

            peoples = []
            for pred, encod, location, rec in zip(self.classifier.predict(faces_encodings), faces_encodings, face_locations, are_matches):
                if rec:
                    people = People(pred)
                else:
                    people = None
                peoples.append((people, location, encod))
            return peoples
        return []


# noinspection SpellCheckingInspection
class NanoFaceDetection(object):
    TEST_DIR = 'tests'
    _classifier = None

    def __init__(self, admines, scale=4):
        self.admines = admines
        self.scale = scale

    @property
    def classifier(self):
        if not self._classifier:
            try:
                with open('%s/classifier.dat' % People.DATA_DIR, 'rb') as f:
                    from pickle import Unpickler
                    self._classifier = Unpickler(f).load()
            except FileNotFoundError:
                from sklearn.neighbors import KNeighborsClassifier
                self._classifier = KNeighborsClassifier(algorithm='ball_tree', weights='distance')
                trainer = Trainer()
                trainer.train(self._classifier)
                with open('%s/classifier.dat' % People.DATA_DIR, 'wb') as f:
                    from pickle import Pickler
                    Pickler(f).dump(self._classifier)
        return self._classifier

    @staticmethod
    def clean_classifier():
        from os import remove
        remove('%s/classifier.dat' % People.DATA_DIR)

    def run(self, test=False, webcam=False):
        print('Khởi động Nano FaceDetection !')
        try:
            if test:
                self.processing_test(self.classifier)
            if webcam:
                self.processing_webcam(self.classifier)
        except KeyboardInterrupt:
            pass

    @staticmethod
    def draw_info(frame, position, name=None, is_warning=False, guest_id=None, time=None, count=None, alerted=False):
        from cv2 import rectangle, putText, FONT_HERSHEY_DUPLEX, FILLED
        (top, right, bottom, left) = position
        title = name
        if not title:
            title = ''
        if guest_id is not None:
            title += ' #%s' % guest_id
        if count is not None and time is not None:
            title += ' %s - %ss' % (count, time)
        if name and not name.startswith('Guest'):
            rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255) if is_warning else (0, 255, 0), FILLED)
            putText(frame, title, (left + 6, bottom - 6), FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
        else:
            rectangle(frame, (left, top), (right, bottom), (255, 0, 0) if alerted else (0, 0, 255), 5)
            rectangle(frame, (left, bottom - 35), (right, bottom), (255, 0, 0) if alerted else (0, 0, 255), FILLED)
            putText(frame, title, (left + 6, bottom - 6), FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    def processing_test(self, classifier):
        from os import environ
        from face_recognition.face_recognition_cli import image_files_in_folder
        from cv2 import imread, resize, imshow, waitKey
        try:
            images = list(filter(
                lambda image_path: image_path.startswith('test') and image_path.endswith('.jpg'),
                image_files_in_folder(self.TEST_DIR)
            ))
        except FileNotFoundError:
            return
        images.sort()
        session = Session(classifier)
        for path in images:
            print('Tìm thấy ảnh %s - Đang thực hiện phân tích ...' % path)
            image = imread(path)
            _image = resize(image, (0, 0), fx=1 / self.scale, fy=1 / self.scale)
            peoples = session.find(_image)
            if len(peoples) > 0:
                for people, location, face_encoding in peoples:
                    top, right, bottom, left = location

                    top *= self.scale
                    right *= self.scale
                    bottom *= self.scale
                    left *= self.scale

                    if people:
                        print('Phát hiện %s trong ảnh %s' % (people.name, path))
                        self.draw_info(image, (top, right, bottom, left), name=people.name)
                    else:
                        self.draw_info(image, (top, right, bottom, left), name='Guest')
                        print('Phát hiện ra người lạ trong ảnh %s' % path)
            else:
                print('Không tìm thấy ai trong ảnh %s' % path)

            if environ.get('DISPLAY'):
                _image = image
                while _image.shape[0] > 1000 or _image.shape[1] > 1000:
                    _image = resize(_image, (0, 0), fx=0.9, fy=0.9)
                imshow('Image ' + path, _image)
                waitKey(1)
        while True:
            if waitKey(1) & 0xFF == ord('q'):
                break

    def send_mail(self, metadata):
        from datetime import datetime
        from mail import SendAlertStranger
        metadata['last_alert'] = datetime.now()
        return SendAlertStranger(self.admines, metadata).start()

    def processing_webcam(self, classifier):
        from os import environ
        from cv2 import VideoCapture, resize, imshow, waitKey
        # from bdc import load_known_faces
        video_capture = VideoCapture(0)
        session = Session(classifier)
        is_empting = False
        # load_known_faces()
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            _frame = resize(frame, (0, 0), fx=1 / self.scale, fy=1 / self.scale)
            _frame = _frame[:, :, ::-1]
            peoples = session.find(_frame)
            if len(peoples) > 0:
                for people, location, face_encoding in peoples:
                    top, right, bottom, left = location

                    top *= self.scale
                    right *= self.scale
                    bottom *= self.scale
                    left *= self.scale

                    face_image = frame[top:bottom, left:right]
                    face_image = resize(face_image, (500, 500))

                    if people:
                        print('Phát hiện %s%s' % (people.name, '( WARNING )' if not people.is_trusted else ''))
                        self.draw_info(frame, (top, right, bottom, left), people.name, is_warning=not people.is_trusted)
                    else:
                        from datetime import datetime
                        from face_recognition import face_locations, face_encodings
                        from bdc import lookup_known_face, register_new_face
                        metadata = lookup_known_face(face_encoding)

                        # If we found the face, label the face with some useful information.
                        if metadata is not None:
                            time_at_door = datetime.now() - metadata['first_seen_this_interaction']
                            time = int(time_at_door.total_seconds())
                        else:
                            metadata = register_new_face(face_encoding, ('Camera', datetime.now(), face_image))
                            time = 0

                        if time % 10:
                            metadata['face_image'] = metadata['face_image'][1:10]
                        if len(metadata['face_image']) < 10:
                            print('Cập nhật ảnh cho %s' % metadata['id'])
                            metadata['face_image'].append(('Camera', datetime.now(), face_image))

                        alerted = False
                        if metadata.get('last_alert'):
                            alerted = (datetime.now() - metadata['last_alert']).total_seconds() < 60
                        else:
                            if time > 10:
                                self.send_mail(metadata)
                                alerted = True

                        print('Phát hiện ra người lạ (id=%s, count=%s, time=%s, alerted=%s)' %
                              (metadata['id'], metadata['seen_count'], time, 'true' if alerted else 'false'))
                        self.draw_info(frame, (top, right, bottom, left), alerted=alerted, guest_id=metadata['id'], count=metadata['seen_count'],
                                       time=time)
                    is_empting = False
            else:
                if not is_empting:
                    print('Không tìm thấy ai trong ảnh')
                is_empting = True
            if environ.get('DISPLAY'):
                imshow('Video', frame)
                waitKey(1)
            if waitKey(1) & 0xFF == ord('q'):
                break

    def __str__(self):
        return 'NanoFaceDetection'


# noinspection SpellCheckingInspection
if __name__ == '__main__':
    from sys import argv

    nano = NanoFaceDetection(admines=[
        ('Chiến', 'nguyenchienbg2k@gmail.com'),
    ])
    # if len(argv) == 1:
    #     exit(nano.run(test=True, webcam=True))
    if len(argv) == 2:
        command = argv.copy().pop()
        if command == 'reset':
            exit(nano.clean_classifier())
        if command == 'test':
            exit(nano.run(test=True))
        # noinspection SpellCheckingInspection
        if command == 'webcam':
            exit(nano.run(webcam=True))
    # noinspection SpellCheckingInspection
    print('Lệnh không hợp lệ !')
    # noinspection SpellCheckingInspection
    print('Cấu trúc: run-face-recognition [reset|test|webcam]')
