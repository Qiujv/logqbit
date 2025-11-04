import sys

from PySide6.QtCore import QSettings
from PySide6.QtGui import Qt
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

ORG, APP = "LogQbit", "ThemeDemo"

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("极简主题切换")

        # 按钮
        self.btn = QPushButton()
        self.btn.clicked.connect(self.toggle)

        # 设置
        central = QWidget(self)
        self.setCentralWidget(central)
        QVBoxLayout(central).addWidget(self.btn)

        # 读取上次选择
        self.settings = QSettings(QSettings.IniFormat, QSettings.UserScope, ORG, APP)
        self.dark = self.settings.value("ui/dark", False, type=bool)
        self.apply(self.dark)

    def apply(self, dark: bool):
        """仅在有 setColorScheme 的平台上生效，其余平台跳过"""
        if hasattr(Qt, "ColorScheme"):          # Qt 6.5+ 才有
            QApplication.instance().styleHints().setColorScheme(
                Qt.ColorScheme.Dark if dark else Qt.ColorScheme.Light)
        self.btn.setText("切换到亮色" if dark else "切换到暗色")

    def toggle(self):
        self.dark = not self.dark
        self.apply(self.dark)
        self.settings.setValue("ui/dark", self.dark)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(300, 150)
    w.show()
    sys.exit(app.exec())