from flask import Flask

sitebokeh = Flask(_name_)

@sitebokeh.route('/')

def hello():
    return 'Hello Flask!'