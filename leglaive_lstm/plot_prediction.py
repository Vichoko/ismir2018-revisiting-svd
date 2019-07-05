from predict import frame_level_predict, init


if __name__ == '__main__':
    args = init()
    y_pred = frame_level_predict(args.model_name, args.input, plot=True)
