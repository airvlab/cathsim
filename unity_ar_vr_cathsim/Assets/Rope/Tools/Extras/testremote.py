from mjremote import mjremote
import time

m = mjremote()
print('Connect: ', m.connect())
b = bytearray(3*m.width*m.height)
t0 = time.time()
for i in range(0,100):
    m.getimage(b)
t1 = time.time()
print('FPS: ', 100/(t1-t0))
m.close()
