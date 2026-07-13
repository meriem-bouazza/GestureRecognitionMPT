from SignalHub.galyQT import MainWindow
from PyQt5.QtCore import Qt

_delete_held = False
_orig_press = MainWindow.keyPressEvent
_orig_release = MainWindow.keyReleaseEvent


def _press(self, event):
    global _delete_held
    if event.key() == Qt.Key_Delete:
        _delete_held = True
    _orig_press(self, event)


def _release(self, event):
    global _delete_held
    if event.key() == Qt.Key_Delete:
        _delete_held = False
    _orig_release(self, event)


MainWindow.keyPressEvent = _press
MainWindow.keyReleaseEvent = _release


def reset_requested():
    return _delete_held
