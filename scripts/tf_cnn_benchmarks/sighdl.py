from __future__ import absolute_import
from __future__ import print_function

import signal

class SignalHandler(object):
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        print("got signal {}".format(signum))
        self.kill_now = True
