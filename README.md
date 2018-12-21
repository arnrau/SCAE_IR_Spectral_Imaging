# SCAE IR Spectral Imaging
This is the README for the paper "Deep representation learning for domain adaptatable classification of infrared spectral imaging data"

In this paper we describe how we deal with strong physical scattering effects using data-intensive pre-training (Rifai et al., 2011). As far as we know this is the first time using FTIR-based digital pathology that does not require correction of the resonant Mie scattering (Bassan et al., 2009/2010) before classification.

First a resource that is needed to run this example:
	https://ruhr-uni-bochum.sciebo.de/s/EgLAt7DieUbjI71

Under this link you will find some data that we can provide without publishing patient data. These are partly the datasets used for pretrainig and finetuning. These tissue samples are commercially available and can therefore be published. The amount of data generated in FTIR imaging experiments often exceeds several GB and often reaches 100 GB per measurement, so that we can only provide a part of the data here.

### Literature
* Contractive Auto-Encoders: Explicit Invariance During Feature Extraction; Rifai et al., 2011
* Resonant Mie scattering (RMieS) correction of infrared spectra from highly scattering biological samples; Bassan et al., 2010
* Resonant Mie scattering in infrared spectroscopy of biological materials–understanding the ‘dispersion artefact; Bassan et al., 2009
* G. van Rossum, Python tutorial, Technical Report CS-R9526, Centrum voor Wiskunde en Informatica (CWI), Amsterdam, May 1995.
* F. Bastien, P. Lamblin, R. Pascanu, J. Bergstra, I. Goodfellow, A. Bergeron, N. Bouchard, D. Warde-Farley and Y. Bengio. “Theano: new features and speed improvements”. NIPS 2012 deep learning workshop. 