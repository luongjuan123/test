import cv2
import face_recognition
import os
import csv
import numpy as np
from datetime import datetime
from PyQt6 import QtWidgets, QtGui, QtCore
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Hàm hỗ trợ
def initialize_attendance(csv_file):
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Student ID", "Name", "Date", "Status", "Note"])

def mark_attendance(student_id, student_name, csv_file):
    today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(csv_file, 'r', newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)
        for row in rows[1:]:
            if row and row[0] == student_id and row[2].startswith(today.split()[0]):
                return False  # Đã điểm danh hôm nay
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([student_id, student_name, today, "1", ""])
    return True

def load_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_ids = []
    known_face_names = []
    known_face_classes = {}
    if not os.path.exists(known_faces_dir):
        os.makedirs(known_faces_dir)
    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg"):
            parts = filename.split('_')
            student_id = parts[0]
            student_name = parts[1]
            class_name = parts[2].split('.')[0] if len(parts) > 2 else "Unknown"
            image_path = os.path.join(known_faces_dir, filename)
            image = face_recognition.load_image_file(image_path)
            # Giảm kích thước ảnh trước khi mã hóa
            image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)  # Giảm 50%
            encoding = face_recognition.face_encodings(image)
            if encoding:
                known_face_encodings.append(encoding[0])
                known_face_ids.append(student_id)
                known_face_names.append(student_name)
                known_face_classes[student_id] = class_name
            del image  # Giải phóng bộ nhớ ngay sau khi dùng
    return known_face_encodings, known_face_ids, known_face_names, known_face_classes

class LoginDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Login")
        self.initUI()

    def initUI(self):
        layout = QtWidgets.QVBoxLayout()
        self.username_input = QtWidgets.QLineEdit()
        self.username_input.setPlaceholderText("Username")
        layout.addWidget(self.username_input)

        self.password_input = QtWidgets.QLineEdit()
        self.password_input.setPlaceholderText("Password")
        self.password_input.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        layout.addWidget(self.password_input)

        self.login_button = QtWidgets.QPushButton("Login")
        self.login_button.clicked.connect(self.login)
        layout.addWidget(self.login_button)

        self.setLayout(layout)

    def login(self):
        username = self.username_input.text()
        password = self.password_input.text()
        # Giả lập tài khoản: admin/admin (có thể thay bằng database sau)
        if username == "admin" and password == "admin":
            self.accept()
        else:
            QtWidgets.QMessageBox.warning(self, "Login Failed", "Invalid username or password")

class AddStudentDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add New Student")
        self.setGeometry(200, 200, 400, 600)
        self.cap = None
        self.image_captured = []
        self.current_angle = 0
        self.angles = ["Straight", "Left", "Right"]
        self.initUI()

    def initUI(self):
        layout = QtWidgets.QVBoxLayout()

        self.id_input = QtWidgets.QLineEdit()
        self.id_input.setPlaceholderText("Enter Student ID")
        layout.addWidget(self.id_input)

        self.name_input = QtWidgets.QLineEdit()
        self.name_input.setPlaceholderText("Enter Student Name")
        layout.addWidget(self.name_input)

        self.class_input = QtWidgets.QLineEdit()
        self.class_input.setPlaceholderText("Enter Class (e.g., 12A1)")
        layout.addWidget(self.class_input)

        self.start_capture_button = QtWidgets.QPushButton("Start Capture")
        self.start_capture_button.clicked.connect(self.start_capture)
        layout.addWidget(self.start_capture_button)

        self.guide_label = QtWidgets.QLabel("Look straight at the camera")
        layout.addWidget(self.guide_label)

        self.video_label = QtWidgets.QLabel()
        self.video_label.setMinimumSize(320, 240)
        layout.addWidget(self.video_label)

        self.capture_button = QtWidgets.QPushButton("Capture")
        self.capture_button.clicked.connect(self.capture_image)
        self.capture_button.setEnabled(False)
        layout.addWidget(self.capture_button)

        self.button_layout = QtWidgets.QHBoxLayout()
        self.confirm_button = QtWidgets.QPushButton("Confirm")
        self.confirm_button.clicked.connect(self.confirm_capture)
        self.confirm_button.setEnabled(False)
        self.button_layout.addWidget(self.confirm_button)

        self.retake_button = QtWidgets.QPushButton("Retake")
        self.retake_button.clicked.connect(self.retake_capture)
        self.retake_button.setEnabled(False)
        self.button_layout.addWidget(self.retake_button)

        layout.addLayout(self.button_layout)
        self.setLayout(layout)

    def start_capture(self):
        if not self.cap:
            self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            if self.id_input.text() in self.parent().known_face_ids:
                QtWidgets.QMessageBox.warning(self, "Duplicate ID", "This Student ID already exists. Please use a different ID.")
                return
            self.start_capture_button.setEnabled(False)
            self.capture_button.setEnabled(True)
            self.confirm_button.setEnabled(False)
            self.retake_button.setEnabled(False)
            self.current_angle = 0
            self.image_captured = []
            self.guide_label.setText(f"Look {self.angles[self.current_angle].lower()} at the camera")
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Giảm kích thước khung hình trước khi xử lý
                frame = cv2.resize(frame, (640, 480))  # Giảm về 640x480
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                # ... (phần còn lại giữ nguyên)
    def capture_image(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                if not face_locations:
                    QtWidgets.QMessageBox.warning(self, "No Face Detected", "No face detected. Please retake this angle.")
                    self.retake_button.setEnabled(True)
                    return
                self.image_captured.append(frame)
                self.current_angle += 1
                if self.current_angle < len(self.angles):
                    self.guide_label.setText(f"Look {self.angles[self.current_angle].lower()} at the camera")
                else:
                    self.timer.stop()
                    self.capture_button.setEnabled(False)
                    self.confirm_button.setEnabled(True)
                    self.retake_button.setEnabled(True)
                    self.guide_label.setText("Captured all angles")
                    self.show_captured_image()

    def show_captured_image(self):
        if self.image_captured:
            frame = self.image_captured[-1]  # Hiển thị ảnh cuối cùng
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            image = QtGui.QImage(rgb_frame.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(image)
            self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio))

    def confirm_capture(self):
        if self.image_captured and self.id_input.text() and self.name_input.text() and self.class_input.text():
            self.student_id = self.id_input.text()
            self.student_name = self.name_input.text()
            self.class_name = self.class_input.text()
            reply = QtWidgets.QMessageBox.question(self, "Confirm Images",
                                                 "Are you satisfied with these images?",
                                                 QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
            if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                for i, frame in enumerate(self.image_captured):
                    img_path = os.path.join("known_faces", f"{self.student_id}_{self.student_name}_{self.class_name}_{i}.jpg")
                    cv2.imwrite(img_path, frame)
                self.cap.release()
                self.accept()

    def retake_capture(self):
        self.start_capture_button.setEnabled(True)
        self.capture_button.setEnabled(False)
        self.confirm_button.setEnabled(False)
        self.retake_button.setEnabled(False)
        self.start_capture()

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        event.accept()

class MarkAttendanceDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mark Attendance")
        self.setGeometry(200, 200, 400, 500)
        self.cap = None
        self.parent = parent
        self.current_students = []
        self.auto_mark = False
        self.last_mark_time = 0
        self.initUI()

    def initUI(self):
        layout = QtWidgets.QVBoxLayout()

        self.video_label = QtWidgets.QLabel()
        self.video_label.setMinimumSize(320, 240)
        layout.addWidget(self.video_label)

        self.student_list = QtWidgets.QListWidget()
        layout.addWidget(self.student_list)

        self.status_label = QtWidgets.QLabel("Press Space or 'Mark' to mark attendance")
        layout.addWidget(self.status_label)

        self.mark_button = QtWidgets.QPushButton("Mark")
        self.mark_button.clicked.connect(self.mark_attendance)
        self.mark_button.setEnabled(False)
        layout.addWidget(self.mark_button)

        self.auto_check = QtWidgets.QCheckBox("Auto Mark (every 3s)")
        self.auto_check.stateChanged.connect(self.toggle_auto_mark)
        layout.addWidget(self.auto_check)

        self.setLayout(layout)
        self.start_camera()

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)  # Tự động dùng camera mặc định
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.warning(self, "Camera Error", "Failed to open default camera.")
            self.reject()
            return
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                self.current_students = []
                self.student_list.clear()
                for i, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):
                    matches = face_recognition.compare_faces(self.parent.known_face_encodings, face_encoding)
                    name, student_id = "Unknown", f"Unknown_{i}"
                    if True in matches:
                        matched_index = matches.index(True)
                        name = self.parent.known_face_names[matched_index]
                        student_id = self.parent.known_face_ids[matched_index]
                    self.current_students.append((student_id, name))
                    self.student_list.addItem(f"{student_id} - {name}")
                    top, right, bottom, left = face_location
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, student_id, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

                self.mark_button.setEnabled(bool(self.current_students))
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                image = QtGui.QImage(rgb_frame.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(image)
                self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio))

                if self.auto_mark and self.current_students:
                    current_time = datetime.now().timestamp()
                    if current_time - self.last_mark_time > 3:
                        self.mark_attendance(auto=True)

    def toggle_auto_mark(self, state):
        self.auto_mark = (state == QtCore.Qt.CheckState.Checked.value)
        self.mark_button.setEnabled(not self.auto_mark or bool(self.current_students))

    def mark_attendance(self, auto=False):
        if self.current_students:
            marked_students = []
            today = datetime.now().strftime("%Y-%m-%d")
            for student_id, student_name in self.current_students:
                if student_id.startswith("Unknown"):
                    continue
                if mark_attendance(student_id, student_name, self.parent.csv_file):
                    marked_students.append(f"{student_name} ({student_id})")
                    self.status_label.setText(f"Marked: {student_name} ({student_id})")
            if marked_students:
                msg = f"Attendance marked for: {', '.join(marked_students)}" if not auto else f"Auto-marked: {', '.join(marked_students)}"
                QtWidgets.QMessageBox.information(self, "Success", msg)
                self.parent.view_attendance()
                self.last_mark_time = datetime.now().timestamp()
                self.parent.send_email_report(today, marked_students)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_Space:
            self.mark_attendance()

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
            self.cap = None  # Đặt lại thành None để tránh tái sử dụng
        event.accept()
class FaceAttendanceApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Attendance System")
        self.setGeometry(100, 100, 1200, 800)
        self.known_faces_dir = "known_faces"
        self.csv_file = "attendance.csv"
        self.known_face_encodings, self.known_face_ids, self.known_face_names, self.known_face_classes = load_known_faces(self.known_faces_dir)
        self.language = "English"
        self.user_role = "admin"  # Giả lập, có thể thêm database sau
        self.initUI()

    def initUI(self):
        if not self.login():
            self.close()
            return

        main_layout = QtWidgets.QHBoxLayout()
        self.setLayout(main_layout)

        # Khung Controls
        control_frame = QtWidgets.QGroupBox("Controls")
        control_layout = QtWidgets.QVBoxLayout()
        control_frame.setLayout(control_layout)

        self.add_student_button = QtWidgets.QPushButton("Add New Student")
        self.add_student_button.clicked.connect(self.add_student)
        control_layout.addWidget(self.add_student_button)

        self.mark_attendance_button = QtWidgets.QPushButton("Mark Attendance")
        self.mark_attendance_button.clicked.connect(self.mark_attendance_dialog)
        control_layout.addWidget(self.mark_attendance_button)

        self.delete_student_button = QtWidgets.QPushButton("Delete Student")
        self.delete_student_button.clicked.connect(self.delete_student)
        control_layout.addWidget(self.delete_student_button)

        self.view_attendance_button = QtWidgets.QPushButton("View Attendance")
        self.view_attendance_button.clicked.connect(self.view_attendance)
        control_layout.addWidget(self.view_attendance_button)

        self.export_button = QtWidgets.QPushButton("Export Report")
        self.export_button.clicked.connect(self.export_report)
        control_layout.addWidget(self.export_button)

        self.stats_button = QtWidgets.QPushButton("View Statistics")
        self.stats_button.clicked.connect(self.view_statistics)
        control_layout.addWidget(self.stats_button)

        self.language_combo = QtWidgets.QComboBox()
        self.language_combo.addItems(["English", "Vietnamese"])
        self.language_combo.currentTextChanged.connect(self.change_language)
        control_layout.addWidget(self.language_combo)

        control_layout.addStretch()

        # Khung bên phải
        right_frame = QtWidgets.QVBoxLayout()

        self.attendance_frame = QtWidgets.QGroupBox("Attendance Records")
        attendance_layout = QtWidgets.QVBoxLayout()
        self.attendance_frame.setLayout(attendance_layout)

        self.class_filter = QtWidgets.QComboBox()
        self.class_filter.addItem("All Classes")
        self.class_filter.addItems(set(self.known_face_classes.values()))
        self.class_filter.currentTextChanged.connect(self.view_attendance)
        attendance_layout.addWidget(self.class_filter)

        self.attendance_table = QtWidgets.QTableWidget()
        self.attendance_table.setMinimumHeight(400)
        attendance_layout.addWidget(self.attendance_table)

        self.note_input = QtWidgets.QLineEdit()
        self.note_input.setPlaceholderText("Enter note for absent students")
        self.note_input.returnPressed.connect(self.add_note)
        attendance_layout.addWidget(self.note_input)

        right_frame.addWidget(self.attendance_frame)

        self.stats_frame = QtWidgets.QGroupBox("Statistics")
        self.stats_layout = QtWidgets.QVBoxLayout()
        self.stats_frame.setLayout(self.stats_layout)
        self.stats_canvas = None
        right_frame.addWidget(self.stats_frame)
        self.stats_frame.hide()

        main_layout.addWidget(control_frame, 1)
        main_layout.addLayout(right_frame, 3)

        self.update_stylesheet()
        initialize_attendance(self.csv_file)
        self.view_attendance()

    def login(self):
        dialog = LoginDialog(self)
        return dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted

    def update_stylesheet(self):
        self.setStyleSheet("""
            QWidget { background-color: #f0f0f0; }
            QGroupBox {
                background-color: #e6f3ff;
                border: 2px solid #4a90e2;
                border-radius: 5px;
                font-weight: bold;
                color: #333;
                margin-top: 15px;
            }
            QGroupBox#attendance_frame {
                background-color: #e6ffe6;
                border: 2px solid #28a745;
            }
            QPushButton {
                background-color: #28a745;
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #218838; }
            QPushButton#add_student_button { background-color: #ff9900; }
            QPushButton#add_student_button:hover { background-color: #e68a00; }
            QPushButton#mark_attendance_button { background-color: #4a90e2; }
            QPushButton#mark_attendance_button:hover { background-color: #357abd; }
            QPushButton#delete_student_button { background-color: #dc3545; }
            QPushButton#delete_student_button:hover { background-color: #c82333; }
            QPushButton#view_attendance_button { background-color: #6f42c1; }
            QPushButton#view_attendance_button:hover { background-color: #5a32a3; }
            QPushButton#export_button { background-color: #17a2b8; }
            QPushButton#export_button:hover { background-color: #138496; }
            QPushButton#stats_button { background-color: #fd7e14; }
            QPushButton#stats_button:hover { background-color: #e06b12; }
            QLabel { font-size: 14px; color: #333; }
            QListWidget { background-color: white; color: #000000; border: 1px solid #ccc; }
            QTableWidget { background-color: white; border: 1px solid #ccc; color: #000000; }
            QTableWidget::item { color: #000000; }
            QHeaderView::section {
                background-color: #d9d9d9;
                color: #000000;
                padding: 5px;
                border: 1px solid #ccc;
                font-weight: bold;
            }
            QLineEdit {
                background-color: white;
                color: #000000;
                border: 1px solid #ccc;
                padding: 5px;
            }
            QComboBox {
                background-color: white;
                color: #000000;
                border: 1px solid #ccc;
                padding: 5px;
            }
            QComboBox::drop-down {
                border-left: 1px solid #ccc;
            }
            QComboBox QAbstractItemView {
                background-color: white;
                color: #000000;
                selection-background-color: #4a90e2;
                selection-color: white;
            }
        """)
        control_frame = self.findChild(QtWidgets.QGroupBox, "control_frame")
        if control_frame:
            control_frame.setObjectName("control_frame")
        self.attendance_frame.setObjectName("attendance_frame")
        self.add_student_button.setObjectName("add_student_button")
        self.mark_attendance_button.setObjectName("mark_attendance_button")
        self.delete_student_button.setObjectName("delete_student_button")
        self.view_attendance_button.setObjectName("view_attendance_button")
        self.export_button.setObjectName("export_button")
        self.stats_button.setObjectName("stats_button")

    def change_language(self, language):
        self.language = language
        translations = {
            "English": {
                "Add New Student": "Add New Student",
                "Mark Attendance": "Mark Attendance",
                "Delete Student": "Delete Student",
                "View Attendance": "View Attendance",
                "Export Report": "Export Report",
                "View Statistics": "View Statistics",
                "Controls": "Controls",
                "Attendance Records": "Attendance Records",
                "Statistics": "Statistics"
            },
            "Vietnamese": {
                "Add New Student": "Thêm Học Sinh Mới",
                "Mark Attendance": "Điểm Danh",
                "Delete Student": "Xóa Học Sinh",
                "View Attendance": "Xem Điểm Danh",
                "Export Report": "Xuất Báo Cáo",
                "View Statistics": "Xem Thống Kê",
                "Controls": "Điều Khiển",
                "Attendance Records": "Hồ Sơ Điểm Danh",
                "Statistics": "Thống Kê"
            }
        }
        tr = translations[self.language]
        self.add_student_button.setText(tr["Add New Student"])
        self.mark_attendance_button.setText(tr["Mark Attendance"])
        self.delete_student_button.setText(tr["Delete Student"])
        self.view_attendance_button.setText(tr["View Attendance"])
        self.export_button.setText(tr["Export Report"])
        self.stats_button.setText(tr["View Statistics"])
        self.findChild(QtWidgets.QGroupBox, "control_frame").setTitle(tr["Controls"])
        self.attendance_frame.setTitle(tr["Attendance Records"])
        self.stats_frame.setTitle(tr["Statistics"])

    def add_student(self):
        if self.user_role != "admin":
            QtWidgets.QMessageBox.warning(self, "Permission Denied", "Only admins can add students.")
            return
        dialog = AddStudentDialog(self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.known_face_encodings, self.known_face_ids, self.known_face_names, self.known_face_classes = load_known_faces(self.known_faces_dir)
            self.class_filter.clear()
            self.class_filter.addItem("All Classes")
            self.class_filter.addItems(set(self.known_face_classes.values()))
            self.view_attendance()

    def mark_attendance_dialog(self):
        dialog = MarkAttendanceDialog(self)
        dialog.exec()

    def delete_student(self):
        if self.user_role != "admin":
            QtWidgets.QMessageBox.warning(self, "Permission Denied", "Only admins can delete students.")
            return
        if not self.known_face_ids:
            QtWidgets.QMessageBox.warning(self, "Warning", "No students to delete.")
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Delete Student")
        layout = QtWidgets.QVBoxLayout()
        label = QtWidgets.QLabel("Select a student to delete:")
        layout.addWidget(label)
        student_combo = QtWidgets.QComboBox()
        for student_id, name in zip(self.known_face_ids, self.known_face_names):
            student_combo.addItem(f"{student_id} - {name}")
        layout.addWidget(student_combo)
        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(lambda: self.confirm_delete(student_combo.currentText(), dialog))
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        dialog.setLayout(layout)
        dialog.exec()

    def confirm_delete(self, selected_student, dialog):
        if not selected_student:
            return
        student_id, name = selected_student.split(" - ", 1)
        reply = QtWidgets.QMessageBox.question(self, "Confirm Deletion",
                                             f"Are you sure you want to delete {name} ({student_id})? This will remove all their attendance data.",
                                             QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            for i in range(3):  # Xóa cả 3 ảnh góc
                img_path = os.path.join(self.known_faces_dir, f"{student_id}_{name}_{self.known_face_classes.get(student_id, 'Unknown')}_{i}.jpg")
                if os.path.exists(img_path):
                    os.remove(img_path)
            self.known_face_encodings, self.known_face_ids, self.known_face_names, self.known_face_classes = load_known_faces(self.known_faces_dir)
            if os.path.exists(self.csv_file):
                with open(self.csv_file, 'r', newline='') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                updated_rows = [row for row in rows if len(row) == 0 or row[0] != student_id]
                with open(self.csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(updated_rows)
            self.class_filter.clear()
            self.class_filter.addItem("All Classes")
            self.class_filter.addItems(set(self.known_face_classes.values()))
            self.view_attendance()
        dialog.accept()

    def view_attendance(self):
        if not os.path.exists(self.csv_file):
            self.attendance_table.clear()
            self.attendance_table.setRowCount(0)
            self.attendance_table.setColumnCount(0)
            return
        with open(self.csv_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            if rows:
                headers = ["Student ID", "Name", "Date", "Status", "Note"]
                self.attendance_table.setColumnCount(len(headers))
                filtered_rows = [rows[0]]
                selected_class = self.class_filter.currentText()
                for row in rows[1:]:
                    student_id = row[0]
                    class_name = self.known_face_classes.get(student_id, "Unknown")
                    if selected_class == "All Classes" or class_name == selected_class:
                        filtered_rows.append(row)
                self.attendance_table.setRowCount(len(filtered_rows) - 1)
                self.attendance_table.setHorizontalHeaderLabels(headers)
                for i, row in enumerate(filtered_rows[1:]):
                    for j, value in enumerate(row):
                        item = QtWidgets.QTableWidgetItem(value)
                        if j == 3:  # Cột Status
                            item.setBackground(QtGui.QColor("#28a745" if value == "1" else "#d9534f"))
                        item.setForeground(QtGui.QColor("#000000"))
                        self.attendance_table.setItem(i, j, item)
                self.attendance_table.resizeColumnsToContents()

    def add_note(self):
        note = self.note_input.text()
        if not note:
            return
        today = datetime.now().strftime("%Y-%m-%d")
        selected_class = self.class_filter.currentText()
        with open(self.csv_file, 'r', newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)
        present_ids = {row[0] for row in rows[1:] if row[2].startswith(today)}
        absent_students = [(sid, sname) for sid, sname in zip(self.known_face_ids, self.known_face_names)
                          if sid not in present_ids and (selected_class == "All Classes" or self.known_face_classes.get(sid, "Unknown") == selected_class)]
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for sid, sname in absent_students:
                writer.writerow([sid, sname, f"{today} 00:00:00", "0", note])
        self.note_input.clear()
        self.view_attendance()

    def export_report(self):
        options = QtWidgets.QFileDialog.options()
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Report", "", "Excel Files (*.xlsx)")
        if not file_name:
            return
        df = pd.read_csv(self.csv_file)
        if file_name.endswith(".xlsx"):
            df.to_excel(file_name, index=False)
        elif file_name.endswith(".pdf"):
            df.to_html("temp.html")
            from weasyprint import HTML
            HTML("temp.html").write_pdf(file_name)
            os.remove("temp.html")
        QtWidgets.QMessageBox.information(self, "Success", f"Report exported to {file_name}")

    def send_email_report(self, date, marked_students):
        sender_email = "your_email@gmail.com"  # Thay bằng email của bạn
        receiver_email = "receiver_email@gmail.com"  # Thay bằng email nhận
        password = "your_app_password"  # Thay bằng mật khẩu ứng dụng Gmail
        subject = f"Attendance Report - {date}"
        body = f"Students marked present on {date}:\n" + "\n".join(marked_students) + "\n\nCheck the full report in the app."
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        try:
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(sender_email, password)
                server.send_message(msg)
        except Exception as e:
            print(f"Failed to send email: {e}")

    def view_statistics(self):
        if not os.path.exists(self.csv_file):
            QtWidgets.QMessageBox.warning(self, "No Data", "No attendance data available.")
            return
        df = pd.read_csv(self.csv_file)
        if df.empty or 'Date' not in df.columns:
            QtWidgets.QMessageBox.warning(self, "No Data", "No valid attendance data available.")
            return
        # Giới hạn chỉ lấy 30 ngày gần nhất
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        df = df[df['Date'] >= (datetime.now().date() - pd.Timedelta(days=30))]
        stats = df.groupby('Date')['Status'].value_counts().unstack().fillna(0)
        stats = stats.reindex(columns=['0', '1'], fill_value=0)
        stats.columns = ['Absent', 'Present']

        self.attendance_frame.hide()
        self.stats_frame.show()
        if self.stats_canvas:
            self.stats_layout.removeWidget(self.stats_canvas)
            self.stats_canvas.deleteLater()

        fig, ax = plt.subplots(figsize=(8, 4))
        stats.plot(kind='bar', stacked=True, ax=ax, color=['#d9534f', '#28a745'])
        ax.set_title("Attendance Statistics")
        ax.set_xlabel("Date")
        ax.set_ylabel("Number of Students")
        plt.xticks(rotation=45)
        self.stats_canvas = FigureCanvas(fig)
        self.stats_layout.addWidget(self.stats_canvas)

    def closeEvent(self, event):
        if self.stats_canvas:
            self.stats_layout.removeWidget(self.stats_canvas)
            self.stats_canvas.deleteLater()
            self.stats_canvas = None
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = FaceAttendanceApp()
    window.show()
    app.exec()