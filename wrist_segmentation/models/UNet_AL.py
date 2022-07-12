from tensorflow.keras.models import *
from tensorflow.keras.layers import *

def model_gen(config):
    def attention_block(input_enc, input_dec, channels, n_filt):
        conv_name = 'AL_conv' + str(n_filt) + '_'
        mult_name = 'AL_mult' + str(n_filt)

        if config.AT_CONV_TYPE == 'depth':
            conv11 = DepthwiseConv2D(1, activation=None, padding='valid', kernel_initializer='he_normal',
                                     strides=(2, 2), depth_multiplier=2, name=conv_name + '11')(input_enc)
            conv12 = DepthwiseConv2D(1, activation=None, padding='valid', kernel_initializer='he_normal',
                                     name=conv_name + '12')(input_dec)
        else:
            conv11 = Conv2D(n_filt, 1, activation=None, padding='valid', kernel_initializer='he_normal',
                            name=conv_name + '11', strides=(2, 2))(input_enc)

            conv12 = Conv2D(n_filt, 1, activation=None, padding='valid', kernel_initializer='he_normal',
                            name=conv_name + '12', strides=(1, 1))(input_dec)

        add = Add()([conv11, conv12])
        relu = ReLU()(add)

        conv2 = Conv2D(1, 1, activation='sigmoid', padding='valid', kernel_initializer='he_normal',
                        name=conv_name + '2')(relu)

        up2 = UpSampling2D(size=2, interpolation="bilinear")(conv2)

        mult = Multiply(name=mult_name)([up2, input_enc])
        return mult

    def encoder_block(input, n_filt, activation, padding, kernel_initializer, channels, pool_size, p_drop):
        conv_name = 'EN_conv' + str(n_filt) + '_'
        conv1 = Conv2D(n_filt, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer,
                       name=conv_name + '1')(input)
        conv2 = Conv2D(n_filt, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer,
                       name=conv_name + '2')(conv1)
        batch = BatchNormalization(axis=channels, name='EN_batch' + str(n_filt))(conv2)
        pool = MaxPooling2D(pool_size=pool_size, name='EN_pool' + str(n_filt))(batch)
        drop = SpatialDropout2D(p_drop, data_format="channels_last", name='EN_drop' + str(n_filt))(pool)
        return [drop, batch]

    def decoder_block(input, input_enc, n_filt, activation, padding, kernel_initializer,
                      channels, pool_size, p_drop):
        conv_name = 'DE_conv' + str(n_filt) + '_'

        up = UpSampling2D(size=pool_size)(input)
        up = Conv2D(n_filt, 2, activation=activation, padding=padding, kernel_initializer=kernel_initializer,
                    name=conv_name + 'up')(up)

        atten = attention_block(input_enc, input, channels, n_filt)

        merge = concatenate([atten, up], axis=channels)

        conv1 = Conv2D(n_filt, 3, activation=activation, padding=padding, name=conv_name + '1',
                       kernel_initializer=kernel_initializer)(merge)
        conv2 = Conv2D(n_filt, 3, activation=activation, padding=padding, name=conv_name + '2',
                       kernel_initializer=kernel_initializer)(conv1)
        batch = BatchNormalization(axis=channels)(conv2)
        return batch

    def bottleneck(input, n_filt, activation, padding, kernel_initializer, channels, pool_size, p_drop):
        conv_name = 'BO_conv' + str(n_filt) + '_'

        conv1 = Conv2D(n_filt, 3, activation=activation, padding=padding, name=conv_name + '1',
                       kernel_initializer=kernel_initializer)(input)
        conv2 = Conv2D(n_filt, 3, activation=activation, padding=padding, name=conv_name + '2',
                       kernel_initializer=kernel_initializer)(conv1)
        batch = BatchNormalization(axis=channels)(conv2)
        drop = SpatialDropout2D(p_drop, data_format='channels_last')(batch)
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
    input_shape = (img_rows, img_cols, 1)

    # Encoder path
    inputs = Input(input_shape, name='input')
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
    outputs = Conv2D(1, 1, activation='sigmoid', name='output')(x)

    model = Model(inputs, outputs, name=config.MODEL_NAME)
    return model
