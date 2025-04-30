# script_generate_install_requires.py

def read_requirements(file_path='requirements.txt'):
    with open(file_path, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return requirements

def print_install_requires(requirements):
    print("Copia esto dentro de 'install_requires' en tu setup.py:\n")
    print("install_requires=[")
    for req in requirements:
        print(f"    '{req}',")
    print("],")

if __name__ == "__main__":
    requirements = read_requirements()
    print_install_requires(requirements)
