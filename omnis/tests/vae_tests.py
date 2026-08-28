import pyro
import pyro.distributions as dist
import torch

from omnis.vae import Decoder, Encoder, VAE


def make_normal(loc, scale):
    return dist.Normal(loc, scale)


def test_vae():
    pyro.clear_param_store()
    torch.manual_seed(42)

    batch_size = 8
    x_dim = 20
    z_dim = 5
    n_z = 3

    encoder = Encoder(
        x_dim=x_dim,
        z_dim=z_dim,
        hidden_dim=32,
    )

    decoder = Decoder(
        z_dim=z_dim,
        x_dim=x_dim,
        hidden_dim=32,
    )

    make_prior = lambda reference: dist.Normal(
        reference.new_zeros(reference.shape[0],z_dim),
        reference.new_ones(reference.shape[0], z_dim),
    )

    vae = VAE(
        p_z=make_prior,
        p_x_given_z=make_normal,
        q_z_given_x=make_normal,
        h_for_p_x=decoder,
        h_for_q_z=encoder,
        z_event_ndims=1,
        x_event_ndims=1,
    )

    x = torch.randn(batch_size, x_dim)

    # 1. Tester l'encodage
    z = vae(x, n_z=n_z)

    assert z.shape == (n_z, batch_size, z_dim)
    assert torch.isfinite(z).all()

    # 2. Tester la reconstruction
    reconstructed_x = vae.reconstruct(
        x,
        n_z=n_z,
    )

    assert reconstructed_x.shape == (
        n_z,
        batch_size,
        x_dim,
    )
    assert torch.isfinite(reconstructed_x).all()

    # 3. Tester la génération
    generated_x = vae.generate(batch_size=batch_size)

    assert generated_x.shape == (batch_size, x_dim)
    assert torch.isfinite(generated_x).all()

    # 4. Tester la perte
    loss = vae.get_training_loss(
        x,
        n_z=n_z,
        objective="elbo",
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    # 5. Tester la rétropropagation
    loss.backward()

    encoder_has_gradients = any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in encoder.parameters()
    )

    decoder_has_gradients = any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in decoder.parameters()
    )

    assert encoder_has_gradients
    assert decoder_has_gradients

def main():
    test_vae()


if __name__ == "__main__":
    main()



