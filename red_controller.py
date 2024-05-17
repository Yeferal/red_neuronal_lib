from neuronal import Neuronal

def create_red_controller():
    inputs = int(input("Numero de entradas: "))
    outputs = int(input("Numero de salidas: "))
    hidden_layers = int(input("Numero de Capas Ocultas: "))
    neurons_hidden_layerd = []
    for i in range(int(hidden_layers)):
        print(f"Capa Oculata {i+1}")
        num_neurons = input("Numero de neurons: ")
        neurons_hidden_layerd.append(int(num_neurons))

    in_new_input = True
    inputs_list = []
    outputs_list = []
    while in_new_input:
        list_x = []
        for e in range(int(inputs)):
            input_x = int(input(f"Entrad x{e+1}: "))
            list_x.append(input_x)
        inputs_list.append(list_x)
        print(f"Salida de la entrada\n{list_x}:\n")

        y_list = []
        for j in range(int(outputs)):
            y = int(input(f"y{i+1} = "))
            y_list.append(y)

        outputs_list.append(y_list)

        is_exit = input("1. Agregar nueva entrada.\n2. Terminar: ")
        if is_exit == "2":
            in_new_input = False

    print("\nTipos de Funciones:")
    type_func_hiddens = input("1. SIGMOID\n2. TAN_H\n")
    type_func_inputs = input("1. STEP\n2. IDENTITY\n")
    type_hiddens = "SIGMOID"
    if type_func_hiddens == "2":
        type_hiddens = "HYPERBOLIC_TANGENT"

    type_out = "STEP"
    if type_func_inputs == "2":
        type_out = "IDENTITY"

    epoch = int(input("Numero de Epoca: "))

    print("CONFIGURACION")
    print("#Entradas:", inputs)
    print("#Salidas:", outputs)
    print("Capa Ocultas: ", neurons_hidden_layerd)
    print("Entradas y Salidas")
    for i in range(len(inputs_list)):
        print(inputs_list[i], outputs_list[i])
    print("TIPO DE FUNCION CAPAS OCULTAS: ", type_hiddens)
    print("TPO DE FUNCION CAPAS SALIDA:", type_out)
    print("epocas: ", epoch)

    init = input("Presiones 0 para empezar: ")
    # Crea la red y empieza
    neuronal = Neuronal(inputs=inputs, outputs=outputs, hidden_layers=hidden_layers,
                        neurons_hidden_layers=neurons_hidden_layerd, hidden_layers_activation_function=type_hiddens,
                        output_layers_activation_function=type_out)

    neuronal.train(inputs_list, outputs_list, epoch)
    print("Prediciones: ")
    for i in range(len(inputs_list)):
        prediction = neuronal.predict(inputs_list[i])
        print(inputs_list[i], "->", prediction)
