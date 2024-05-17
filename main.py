from red_controller import create_red_controller


def main():
    is_exit = False
    while not is_exit:
        print("OPCIONES: ")
        print("1. Ingresa nueva red nueronal")
        print("0. Salir de la aplicacion")
        option = int(input("Opcion: "))
        if option == 1:
            create_red_controller()
        elif option == 0:
            is_exit = True


main()