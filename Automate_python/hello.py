"""
This is purely module, could be used only under __name__ == "__main__"
entry
"""
# def hello():
#    print("module hello world")
#    return

"""
Adding __name__ == "__main__" will force the command to run only
under those line above 
"""
# if __name__ == "__main__":
#    print("Main Hello World")

"""
This without __name__ == "__main__" entry point 
Line below could ran by either
python3 hello.py
or
python3 -m hello
"""
print("Hello World")
