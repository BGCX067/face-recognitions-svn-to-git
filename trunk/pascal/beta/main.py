# -*- coding: utf-8 -*-
"""
Bootstrap-Modul zum Starten der Anwendung.

"""
import sys
import logging as log

from PyQt4 import Qt
from PyQt4 import QtGui

import gui

def main(args):    
    """Hauptfenster, Hauptanwendung Initialisierung und Schliessen Signal anbinden""" 
    # logger der gesamten Anwendung, DEGUB->Details, INFO->nur infos, CRITICAL->nur Errors
    log.basicConfig(format='%(levelname)s: %(message)s', level=log.DEBUG) # INFO, DEBUG, CRITICAL
    log.info('START der Anwendung')

    app = QtGui.QApplication(args)
    win = gui.GUI()
    win.show()
    app.connect(app,                             # Sender
                Qt.SIGNAL('lastWindowClosed()'), # Signal
                app,                             # Empfaenger
                Qt.SLOT('quit()')                # aktivierter Slot
                )
    app.connect(app,                             # Sender
            Qt.SIGNAL('lastWindowClosed()'), # Signal
            win.video.controller.on_close                # aktivierter Slot
            )
    return app.exec_()
    
if __name__ == '__main__':
    # Endlosschleifen aufruf, app.exec_ als returnwert
    sys.exit(main(sys.argv))
    