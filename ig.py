import tensorflow as tf
from settings import *

def interpolate_images(baseline, image, alphas):
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(image, axis=0)
    delta = input_x - baseline_x
    images = baseline_x +  alphas_x * delta
    return images


def compute_gradients(model, images, target_class_idx):
    with tf.GradientTape() as tape:
        tape.watch(images)
        logits = model(images)
        probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
    return tape.gradient(probs, images)


def integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients


def integrated_gradients(model, baseline, image, target_class_idx, m_steps=interp_steps, batch_size=8):
    # Generate alphas.
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)
    # Collect gradients
    gradient_batches = tf.TensorArray(tf.float32, size=m_steps+1)
    for alpha in tf.range(0, len(alphas), batch_size):
        from_ = alpha
        to = tf.minimum(from_ + batch_size, len(alphas))
        alpha_batch = alphas[from_:to]
        # Interpolate original image with baseline images
        interpolated_path_input_batch = interpolate_images(baseline=baseline,
                                                           image=image,
                                                           alphas=alpha_batch)
        # Compute gradients between model outputs and interpolated inputs.
        gradient_batch = compute_gradients(model=model, 
                                           images=interpolated_path_input_batch,
                                           target_class_idx=target_class_idx)
        # Write batch indices and gradients to extend TensorArray.
        gradient_batches = gradient_batches.scatter(tf.range(from_, to), gradient_batch)    

    # Stack path gradients together row-wise into single tensor.
    total_gradients = gradient_batches.stack()

    # Integral approximation through averaging gradients.
    avg_gradients = integral_approximation(gradients=total_gradients)
    # Scale integrated gradients with respect to input.
    integrated_gradients = (image - baseline) * avg_gradients
    # Generate mask from aggregating gradients
    mask = tf.reduce_sum(tf.math.abs(integrated_gradients), axis=-1) 

    return mask