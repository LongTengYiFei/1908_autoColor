# 这个脚本是一些函数，不能直接运行

def loss_name_select():
    """
    这个函数可以用来选择损失函数，而且不用传入参数
    """
    print("1: mean_squared_error")
    print("2: mean_absolute_error")
    print("3: mean_absolute_percentage_error")
    print("4: mean_squared_logarithmic_error")
    print("5: squared_hinge")
    print("6: hinge")
    print("7: categorical_hinge")
    print("8: logcosh")
    print("9: categorical_crossentropy")
    print("10: sparse_categorical_crossentropy")
    print("11: binary_crossentropy")
    print("12: kullback_leibler_divergence")
    print("13: poisson")
    print("14: cosine_proximity")
    print("#####以上是可以选择的损失函数.")

    loss_index = input("! ! !你想选择什么损失函数？键入序号按回车结束：")
    loss_index = int(loss_index)

    if(loss_index == 1):
        return "mean_squared_error"
    if (loss_index == 2):
        return "mean_absolute_error"
    if (loss_index == 3):
        return "mean_absolute_percentage_error"
    if (loss_index == 4):
        return "mean_squared_logarithmic_error"
    if (loss_index == 5):
        return "squared_hinge"
    if (loss_index == 6):
        return "hinge"
    if (loss_index == 7):
        return "categorical_hinge"
    if (loss_index == 8):
        return "logcosh"
    if (loss_index == 9):
        return "categorical_crossentropy"
    if (loss_index == 10):
        return "sparse_categorical_crossentropy"
    if (loss_index == 11):
        return "binary_crossentropy"
    if (loss_index == 12):
        return "kullback_leibler_divergence"
    if (loss_index == 13):
        return "poisson"
    if (loss_index == 14):
        return "cosine_proximity"

def optimizer_name_select():

    """
     这个函数可以用来选择优化器，而且不用传入参数
        """
    print("1: SGD")
    print("2: RMSprop")
    print("3: Adagrad")
    print("4: Adadelta")
    print("5: Adam")
    print("6: Adamax")
    print("7: Nadam")
    print("#####以上是可以选择的优化器.")

    loss_index = input("! ! !你想选择什么优化器？键入序号按回车结束：")
    loss_index = int(loss_index)

    if (loss_index == 1):
        return "SGD"
    if (loss_index == 2):
        return "RMSprop"
    if (loss_index == 3):
        return "Adagrad"
    if (loss_index == 4):
        return "Adam"
    if (loss_index == 5):
        return "Adamax"
    if (loss_index == 6):
        return "Nadam"

