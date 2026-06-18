import threading
import pyttsx3
import traceback

def w():
    try:
        import pythoncom
        pythoncom.CoInitialize()
        engine = pyttsx3.init()
        engine.say('test')
        engine.runAndWait()
        print('Success in thread')
    except Exception as e:
        print('Fail in thread:', e)
        traceback.print_exc()

t = threading.Thread(target=w)
t.start()
t.join()
