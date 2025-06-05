import socket

with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
    s.connect("/tmp/VLiDAR.sock")
    print("Connected to the socket")
    # s.sendall(b"Hello, world")
    while True:
        data = s.recv(1024)
        if not data:
            break
        print(f"Received {data!r}")

print(f"Received {data!r}")
