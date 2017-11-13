from Kernel import Kernel, Ckfunction, cs
import numpy as np
from IPython import embed
from MyMathApp.mathobjects.ExpressionManager import Expressions

class PeriodicKernel(Kernel):
	"""docstring for PeriodicKernel"""
	def __init__(self, indexshape=(1,1),outshape=(1,1),symtype=cs.SX):
		super(PeriodicKernel, self).__init__(indexshape=indexshape,outshape=(1,1),symtype=symtype)
		self.Expressions = Expressions()
		lam0 = symtype.sym('lam0',1,1)
		rlam = symtype.sym('rlam',1,1)
		p = symtype.sym('period',1,1)
		self.Expressions.add(member='Params',name='lam0',expr=lam0,eq=cs.DM(1.))
		self.Expressions.add(member='Params',name='rlam',expr=rlam,eq=cs.DM(0.1))
		self.Expressions.add(member='Params',name='period',expr=p,eq=cs.DM(1.))
		e = (self.xsym-self.ysym)/p;
		ker = (lam0**2)*(1-rlam*cs.cos(e))/(1+rlam**2-2*rlam*cs.cos(e))
		self.kernel = Ckfunction('per_kernel',[self.xsym,self.ysym,self.Expressions.flatten('Params','expr')],[ker])
		self.__paramsval__ = self.Expressions.flatten('Params','val')
		self.__params__ = self.Expressions.flatten('Params','expr')

	def paramupdate(self):
		self.__paramsval__ = self.Expressions.flatten('Params','val')
		self.__params__ = self.Expressions.flatten('Params','expr')