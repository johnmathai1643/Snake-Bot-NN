import io
import os
import socket
import struct
import cv2
import numpy
import pygame

# Start a socket listening for connections on 0.0.0.0:8000 (0.0.0.0 means
# all interfaces)
server_socket = socket.socket()
server_socket.bind(('0.0.0.0', 8000))
server_socket.listen(0)
i = 0
# Accept a single connection and make a file-like object out of it
connection = server_socket.accept()[0].makefile('rb')
try:
    while True:
        # Read the length of the image as a 32-bit unsigned int. If the
        # length is zero, quit the loop
        image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
        if not image_len:
            break
        # Construct a stream to hold the image data and read the image
        # data from the connection
        image_stream = io.BytesIO()
        image_stream.write(connection.read(image_len))
        # Rewind the stream, open it as an image with opencv and do some
        # processing on it
        image_stream.seek(0)
	data = numpy.fromstring(image_stream.getvalue(), dtype=numpy.uint8)
	# "Decode" the image from the array, preserving colour
	image_cv = cv2.imdecode(data, 1)
	# image_cv = cv2.cvtColor(image_cv,image_cv, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(i)+".jpg", image_cv)
	i = i + 1
        print('Image is verified')
finally:
    connection.close()
    server_socket.close()
