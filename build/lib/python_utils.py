import tomllib

def load_paths():
    with open('paths.toml', 'rb') as file:
        paths = tomllib.load(file)
    return paths