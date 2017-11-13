from MyMathApp.mathobjects.Kernels.SqExp import SqExp
from Kernel import Kernel, Ckfunction, cs
import numpy as np
from IPython import embed
class MultioutSqExp(Kernel):
	"""docstring for MultioutKernel"""
	def __init__(self,indexshape=(1,1),outshape=(2,1),symtype=cs.SX):
		super(MultioutSqExp, self).__init__(indexshape,outshape,symtype)
		k1 = SqExp(indexshape);
		ker = k1*np.array([[2.,1.],[1.,3.]]) # dummy kernel
		self.__params__ = ker.__params__
		self.__paramsval__ = ker.__paramsval__
		self.kernel = ker.kernel

if __name__ == '__main__':
	

	K = MultioutSqExp()

	x1 = cs.SX.sym('x1',1,3);
	x2 = cs.SX.sym('x2',1,2);

	y = K(x1,x2)

	print(y.shape)
		