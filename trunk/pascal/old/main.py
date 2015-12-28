# -*- coding: utf-8 -*-
"""
Bootstrap Modul zum Starten der Anwendung.

"""
import sys

from PyQt4 import Qt
from PyQt4 import QtGui

import gui
import model

def main(args):
    """Hauptfenster, Hauptanwendung Initialisierung und Schliessen Signal anbinden""" 
    app = QtGui.QApplication(args)
    win = gui.GUI()
    win.show()
    app.connect(app,                             # Sender-Widget
                Qt.SIGNAL('lastWindowClosed()'), # Signal
                app,                             # Empfaenger
                Qt.SLOT('quit()')                # aktivierter Slot
                )
    return app.exec_()
    
if __name__ == '__main__':
    # Endlosschleifen aufruf app.exec_ als returnwert
    sys.exit(main(sys.argv))