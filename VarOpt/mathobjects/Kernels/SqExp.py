import casadi as cs
from numpy import tril_indices
from Kernel import Kernel, Ckfunction
from IPython import embed
# embed()

class WeakInf(float):
	"""docstring for WeakInf"""
	def __new__(cls):
		obj = super(WeakInf, cls).__new__(cls,cs.inf)
		return obj
		# self = cs.inf
	def __Mul__(self,other):
		if other == 0:
			return 0.
		elif other == cs.inf:
			return cs.inf
		else:
			return WeakInf()

	def __mul__(self,other):
		return self.__Mul__(other)

	def __rmul__(self,other):
		return self.__Mul__(other)
# embed()

class SqExp(Kernel):
	"""docstring for SqExp"""
	def triltoflat(self,Q):
		N = Q.shape[0];
		cla = Q.__class__
		ind = tril_indices(N);
		l = cla.zeros(N*(N+1)/2,1)
		for i in range(ind[0].shape[0]):
			l[i,0] = Q[ind[0][i],ind[1][i]];
		return l

	def __init__(self,indexshape=(1,1),symtype=cs.SX):
		super(SqExp, self).__init__(indexshape=indexshape,outshape=(1,1),symtype=symtype)
		self.scaling = symtype.sym('asqr',1);
		self.cholinv = cs.tril(symtype.sym('linv',indexshape[0],indexshape[0]))
		self.__params__ = cs.vertcat(self.scaling,self.triltoflat(self.cholinv))
		self.__paramsval__ = cs.DM.ones(self.__params__.shape)
		r = self.xsym-self.ysym;
		e = cs.mtimes(self.cholinv,r);
		ker = self.scaling*cs.exp(-0.5*cs.mtimes(e.T,e))
		self.kernel = Ckfunction('kernel',[self.xsym,self.ysym,self.__params__],[ker])
	
	def setinverselengthscale(self,Linv):
		# embed()
		self.__paramsval__[1:] = self.triltoflat(cs.tril(Linv))

	def setvar(self,var):
		self.__paramsval__[0] = var




