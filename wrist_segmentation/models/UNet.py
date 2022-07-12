from tensorflow.keras.models import *
from tensorflow.keras.layers import *

def model_gen(config):
    def encoder_block(input, n_filt, activation, padding, kernel_initializer, channels, pool_size, p_drop):
        conv_name = 'EN_conv' + str(n_filt) + '_'
        conv1 = Conv2D(n_filt, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer,
                       name=conv_name + '1')(input)
        conv2 = Conv2D(n_filt, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer,
                       name=conv_name + '2')(conv1)
        batch = BatchNormalization(axis=channels, name='EN_batch' + str(n_filt))(conv2)
        pool = MaxPooling2D(pool_size=pool_size, name='EN_pool' + str(n_filt))(batch)
        drop = Dropout(p_drop)(pool)
        return [drop, batch]

    def decoder_block(input, input_enc, n_filt, activation, padding, kernel_initializer,
                      channels, pool_size, p_drop):
        conv_name = 'DE_conv' + str(n_filt) + '_'

        up = UpSampling2D(size=pool_size)(input)
        up = Conv2D(n_filt, 2, activation=activation, padding=padding, kernel_initializer=kernel_initializer,
                    name=conv_name + 'up')(up)

        merge = concatenate([input_enc, up], axis=channels)

        conv1 = Conv2D(n_filt, 3, activation=activation, padding=padding, name=conv_name + '1',
                       kernel_initializer=kernel_initializer)(merge)
        conv2 = Conv2D(n_filt, 3, activation=activation, padding=padding, name=conv_name + '2',
                       kernel_initializer=kernel_initializer)(conv1)
        batch = BatchNormalization(axis=channels)(conv2)
        drop = Dropout(p_drop)(batch)
        return drop

    def bottleneck(input, n_filt, activation, padding, kernel_initializer, channels, pool_size, p_drop):
        conv_name = 'BO_conv' + str(n_filt) + '_'

        conv1 = Conv2D(n_filt, 3, activation=activation, padding=padding, name=conv_name + '1',
                       kernel_initializer=kernel_initializer)(input)
        conv2 = Conv2D(n_filt, 3, activation=activation, padding=padding, name=conv_name + '2',
                       kernel_initializer=kernel_initializer)(conv1)
        batch = BatchNormalization(axis=channels)(conv2)
        drop = Dropout(p_drop)(batch)
        return drop

    # Config
    activation = config.ACTIVATION
    padding = config.PADDING
    n_filt = config.FILTERS
    noise = config.NOISE_LEVEL
    p_drop = config.PDROP
    kernel_initializer = config.K_INITIALIZER
    channels = config.CHANNELS
    pool_size = config.POOL_SIZE

    # Start model building
    img_rows, img_cols = config.IMAGE_SIZE
    input_shape = (img_rows, img_cols, None)

    # Encoder path
    inputs = Input(input_shape)
    x = GaussianNoise(noise)(inputs)
    for_AL = []
    filt = n_filt
    for i in range(config.N_DOWN_LAYERS):
        [x, y] = encoder_block(x, filt, activation, padding, kernel_initializer, channels, pool_size, p_drop)
        for_AL.append(y)
        filt = filt * 2

    # Bottleneck
    bottle = bottleneck(x, filt, activation, padding, kernel_initializer, channels, pool_size, p_drop)

    # Decoder path
    x = bottle
    filt = filt // 2
    for i in range(config.N_UP_LAYERS):
        x = decoder_block(x, for_AL[-1 - i], filt, activation, padding, kernel_initializer, channels, pool_size, p_drop)
        filt = filt // 2

    x = Conv2D(2, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization(axis=channels)(x)
    outputs = Conv2D(1, 1, activation='sigmoid')(x)

    model = Model(inputs, outputs, name=config.MODEL_NAME)
    return model