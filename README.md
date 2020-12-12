# m2caiSeg: Semantic Segmentation of Laparoscopic Images using Convolutional Neural Networks

Code for our paper [m2caiSeg: Semantic Segmentation of Laparoscopic Images using Convolutional Neural Networks](https://arxiv.org/abs/2008.10134)

The dataset is available on kaggle at: https://www.kaggle.com/salmanmaq/m2caiseg

If you use our code or data in your work, please do cite our paper

```
@article{maqbool2020m2caiseg,
  title={m2caiSeg: Semantic Segmentation of Laparoscopic Images using Convolutional Neural Networks},
  author={Maqbool, Salman and Riaz, Aqsa and Sajid, Hasan and Hasan, Osman},
  journal={arXiv preprint arXiv:2008.10134},
  year={2020}
}
```

I know this is quite disorganized, and possibly a bit oudated and not maintained. I do not plan to refactor it; really sorry about it as I have become really occupied with other tasks. However, if you require any help in running this, or run into any issues, please create an issue here on GitHub and I would try to help resolve it as much as I can. I would also encourage just using the dataset in a better framework or library if you just need to use the dataset.

This is essentially a Convolutional Encoder Decoder network based on the SegNet architecture for unsupervised feature learning. The particular network can be used for unsupervised feature learning on particular datasets, as well as then fine-tune (further train) the pre-trained network for semantic segmentation.
The PyTorch Dataset class for the m2caiSeg dataset has been provided. You can make the changes in that to adapt to your dataset.

A sample training script is also provided.


\
 \
 \
 \
 \
 \
 \
 \
 \
**IGNORE BELOW - IT IS OUTDATED**
### Deprecated readme - please ignore
[ToDo]
1. Update the Readme 
  -> Provide detailed instructions on how to run and where to make changes
  -> List dependencies
2. Use a different loss than MSELoss
3. Remove unnecessary code/comments and add more helpful comments
4. Implement code to train and evaluate on part of dataset (in between epochs) and make it a parameter
5. Try more transforms
6. Make displaying images a parameter
7. Nice graphs -> Possibly use Visdom or Tensorboard
