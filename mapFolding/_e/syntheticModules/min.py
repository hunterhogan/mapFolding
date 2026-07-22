from __future__ import annotations
BK=isinstance
BJ=property
Av='extractUndeterminedPiles'
AT=slice
AE=list
A6=frozenset
A4=enumerate
A3=sum
x=map
w=dict
o='0'
n=sorted
k=set
d=filter
T=len
K=False
E=True
N=tuple
A=None
I=int
G=range
from bisect import bisect_right as BL
from collections import Counter,defaultdict as BM,deque as U
from concurrent.futures import as_completed as AU,ProcessPoolExecutor as AV
from functools import cache as B,partial as AF,reduce as Aw
from gmpy2 import bit_clear as Ak,bit_flip as f,bit_mask as X,bit_scan1 as Ax,bit_set as Ay,bit_test as O,f_mod_2exp as BN,is_even as g,is_odd as b,mpz,xmpz
from humpy_cytoolz import assoc as BO,compose as Al,concat,curry as y,dissoc as BP,first,get,groupby as AW,itemfilter as BQ,keyfilter as A7,merge as Az,unique as A_,valfilter as A8,valfilter as BR,valfilter as BS
from hunterMakesPy import decreasing as c,errorL33T as B0,inclusive as L,raiseIfNone as P,zeroIndexed as AK
from hunterMakesPy.parseParameters import defineConcurrencyLimit as Am,intInnit as BT
from itertools import accumulate as An,chain as B1,combinations as AL,filterfalse as BU,product as B2
from math import factorial as BV,log,prod
from more_itertools import all_unique as Ao,iter_index as BW,last,loops as B3,one,pairwise as BX,partition as BY,triplewise as BZ
from operator import add as A9,attrgetter as Ba,getitem as V,itemgetter as Bb,methodcaller as AM,mul,neg as Y,sub
from sys import maxsize as Bc
from tqdm import tqdm as AX
from typing import cast,overload as AG,TYPE_CHECKING as Bd,TypeAlias
from Z0Z_tools import between as AN,consecutive as AY,DOTitems as h,DOTkeys as AZ,DOTvalues as Aa,exclude as p,reverseLookup as Ab,thisHasThat as Be,thisNotHaveThat as B4
import dataclasses as l
if Bd:from collections.abc import Callable,Iterable,Iterator,Sequence;from concurrent.futures import Future;from hunterMakesPy import CallableFunction;from hunterMakesPy.theTypes import Limitation;from typing import Self,TypeIs
@B
def B5(mapShape):
	C=mapShape;A=1
	for B in C:
		if B>Bc//A:D=f"I received `dimension = {B!r}` in `mapShape = {C!r}`, but the product of the dimensions exceeds the maximum size of an integer on this system.";raise OverflowError(D)
		A*=B
	return A
def Ac(CPUlimit,concurrencyPackage=A):
	D=concurrencyPackage;C=CPUlimit
	if D=='numba':from numba import get_num_threads as E,set_num_threads as F;B=Am(limit=C,cpuTotal=E());F(B);B=E()
	elif D in{'multiprocessing',A}:B=Am(limit=C)
	else:B=Am(limit=C)
	return B
type Ck=I
AO=I
Ad=mpz
type Bf=AO|Ad
type AP=I
type B6=N[AO,...]
type Cl=w[AP,AO]
type Cm=w[AP,Ad]
z=0
Ae=0
def i(state,leaf):A=state;return CJ(leaf,A.dimensionsTotal,A.mapShape,A.leavesTotal)
def Bg(state,pile):A=state;return Ce(pile,A.dimensionsTotal,A.mapShape,A.leavesTotal)
def B7(state):A=state;return{B:Bg(A,B)for B in G(A.leavesTotal)}
def AH(leafOptions):A=xmpz(leafOptions);A[-1]=0;return A.iter_set()
def A0(leavesTotal,leaves):return Aw(Ak,leaves,X(leavesTotal+L))
def Bh(leavesTotal,leaves):return Aw(Ay,leaves,Ay(0,leavesTotal))
def Bi(leafOptions):return leafOptions.bit_count()-1
def Bj(B):
	C=B
	if A1(B):
		if B.bit_count()==2:C=P(B.bit_scan1())
		elif B.bit_count()==1:C=A
	return C
@y
def Bk(leafOptionsDISPOSABLE,leafOptions):return leafOptions&leafOptionsDISPOSABLE
def B8(mapShape):return N(An(mapShape,mul,initial=1))
def Bl(mapShape):return N(An(B8(mapShape),A9,initial=0))
def AA(productsOfDimensions,dimensionsTotal=A,dimensionFrom首=A):
	D=productsOfDimensions;C=dimensionFrom首;B=dimensionsTotal;B=B or T(D)-1
	if C is A:C=B
	E=C-(B+AK);F=D[0:E][::-1];G=N(An(F,A9,initial=0));return G
def Cn(mapShape):A=mapShape;return d(lambda indices:1<T(indices),x(N,x(AF(BW,A),A_(d((1).__lt__,A)))))
class j(w[AP,Bf]):
	def addMissingPileLeafSpace(A,missing):A=j(n(h(Az(missing,A,factory=j))));return A.copy()
	def atPilePinLeaf(A,pile,leaf):return j(BO(A,pile,leaf,j))
	def atPilePinLeafSafetyFilter(A,pile,leaf):return A.leafPinnedAtPile吗(leaf,pile)or A.pileUndetermined吗(pile)and A.leafNotPinned吗(leaf)
	def bifurcate(A):B=A.extractPinnedLeaves();return B,cast('UndeterminedPiles',BP(A,*AZ(B)))
	def copy(A):return j(A)
	def deconstructAtPile(B,pile=A,leavesToPin=()):
		D=leavesToPin;C=pile
		if C is A:C=first(A8(A1,B))
		if(F:=B.getLeafOptions(C))is A:E=U([B])
		else:D=D or AH(F);E=x(AF(B.atPilePinLeaf,C),d(B.leafNotPinned吗,D))
		return E
	def deconstructByDomainOfLeaf(A,leaf,leafDomain):
		B=leaf;C=U()
		if A.leafNotPinned吗(B):D=Al(Af(B),AF(A.getLeafOptions,default=X(T(A))));E=AF(A.atPilePinLeaf,leaf=B);C.extend(x(E,d(D,d(A.pileUndetermined吗,leafDomain))))
		else:C.append(A)
		return C
	def deconstructByDomainsCombined(A,leaves,leavesDomain):
		D=leavesDomain;C=leaves;E=U()
		def H(index):
			def B(domain):return A.pileUndetermined吗(domain[index])
			return B
		def I(index):
			B=index
			def D(domain):D=P(A.getLeafOptions(domain[B],default=X(T(A))));return Af(C[B],D)
			return D
		def J(leaf,index):
			def B(domain):return A.leafPinnedAtPile吗(leaf,domain[index])
			return B
		if any(x(A.leafNotPinned吗,C)):
			for B in G(T(C)):
				if A.leafNotPinned吗(C[B]):D=d(H(B),D);D=d(I(B),D)
				else:D=d(J(C[B],B),D)
			for K in D:
				F=A.copy()
				for B in G(T(C)):F=F.atPilePinLeaf(K[B],C[B])
				E.append(F)
		else:E.append(A)
		return E
	def extractPinnedLeaves(A):return w(n(h(A8(r,A))))
	def extractUndeterminedPiles(A):return w(n(h(A8(A1,A))))
	@AG
	def getLeaf(self,pile,default=A):...
	@AG
	def getLeaf(self,pile,default):...
	@AG
	def getLeaf[个](self,pile,default):...
	def getLeaf[个](B,pile,default=A):
		A=B.get(pile)
		if r(A):return A
		return default
	@AG
	def getLeafOptions(self,pile,default=A):...
	@AG
	def getLeafOptions(self,pile,default):...
	@AG
	def getLeafOptions[个](self,pile,default):...
	def getLeafOptions[个](B,pile,default=A):
		A=B.get(pile)
		if A1(A):return A
		return default
	def leafNotPinned吗(A,leaf):return leaf not in A.values()
	@BJ
	def leafCount(self):return A3(x(r,self.values()))
	def leafPinned吗(A,leaf):return leaf in A.values()
	def leafPinnedAtPile吗(A,leaf,pile):return leaf==A.get(pile)
	def makeFolding(A,leavesToInsert=()):B=AZ(A.extractUndeterminedPiles());return N(Aa(w(n(h(cast('PinnedLeaves',Az(A,w(zip(B,leavesToInsert,strict=E)),factory=j)))))))
	def pilePinned吗(A,pile):return r(A[pile])
	def pileUndetermined吗(A,pile):return not r(A[pile])
@l.dataclass(slots=E)
class t:
	mapShape:N[I,...]=l.field(init=E);groupsOfFolds:I=0;listFolding:U[B6]=l.field(default_factory=U[B6],init=E);listPermutationSpace:U[j]=l.field(default_factory=U[j],init=E);pile:AP=-1;permutationSpace:j=l.field(default_factory=j,init=E);Theorem2aMultiplier:I=1;Theorem2Multiplier:I=1;Theorem3Multiplier:I=1;Theorem4Multiplier:I=1;dimensionsTotal:I=l.field(init=K);foldingCheckSum:I=l.field(init=K);leafLast:AO=l.field(init=K);leavesTotal:I=l.field(init=K);pileLast:AP=l.field(init=K);pilesTotal:I=l.field(init=K);productsOfDimensions:N[I,...]=l.field(init=K);sumsOfProductsOfDimensions:N[I,...]=l.field(init=K);sumsOfProductsOfDimensionsNearest首:N[I,...]=l.field(init=K);首:I=l.field(init=K)
	@BJ
	def foldsTotal(self):A=self;return prod((A.groupsOfFolds,A.Theorem2aMultiplier,A.Theorem2Multiplier,A.Theorem3Multiplier,A.Theorem4Multiplier))
	def __post_init__(A):
		A.dimensionsTotal=T(A.mapShape);A.leavesTotal=B5(A.mapShape)
		if 0<A.leavesTotal:A.Theorem2aMultiplier=A.leavesTotal
		A.leafLast=A.leavesTotal-1;A.foldingCheckSum=A.leafLast*A.leavesTotal//2;A.pilesTotal=A.leavesTotal;A.pileLast=A.pilesTotal-1;A.首=A.leavesTotal;A.productsOfDimensions=B8(A.mapShape);A.sumsOfProductsOfDimensions=Bl(A.mapShape);A.sumsOfProductsOfDimensionsNearest首=AA(A.productsOfDimensions,A.dimensionsTotal,A.dimensionsTotal)
	def moveToListFolding(A):B=AW(Al(A.leavesTotal.__eq__,Ba('leafCount')),A.listPermutationSpace);A.listPermutationSpace=U(B.get(K,()));A.listFolding.extend(x(AM('makeFolding'),B.get(E,())));return A
	def permutationSpaceCreaseViolation吗(A,permutationSpace):
		B=permutationSpace;J={B:A for(A,B)in h(B.extractPinnedLeaves())}
		for C in G(A.dimensionsTotal):
			D=[[],[]]
			for(L,F)in B.extractPinnedLeaves().items():
				H=As(A.mapShape,F,C)
				if H:
					I=J.get(H)
					if I:D[At(A.mapShape,F,C)].append((L,I))
			for M in D:
				if any(Aj(A,C,B,D)for((A,B),(C,D))in AL(n(M),2)):return E
		return K
	def pinAt_pile吗(A,leaf):return all((A.permutationSpace.leafNotPinned吗(leaf),A.permutationSpace.pileUndetermined吗(A.pile),A.pile in i(A,leaf)))
	def reduceAllPermutationSpace(B,listFunctionsReduction):
		F=listFunctionsReduction;G=B.listPermutationSpace;B.listPermutationSpace=U();H=U()
		while G:
			A=G.pop();I=A3(A.values());C=U(F);D=E
			while D:
				J=C.popleft();A=J(B,P(A))
				if not A:D=K
				elif I!=A3(A.values()):C=U(F);I=A3(A.values())
				elif not C:H.append(A);D=K
		else:B.listPermutationSpace.extend(H)
		return B
	def removeCreaseViolations(A):B=A.listPermutationSpace.copy();A.listPermutationSpace=U();A.listPermutationSpace.extend(BU(A.permutationSpaceCreaseViolation吗,B));return A
@y
def Af(leaf,leafOptions):return leafOptions.bit_test(leaf)
@y
def B9(leavesPinned,leaf):return leaf in leavesPinned.values()
@y
def BA(pileLast,pile):return pileLast!=pile
def r(leafSpace):return BK(leafSpace,AO)
def A1(leafSpace):return BK(leafSpace,Ad)
def Bm(listPermutationSpace,leaf,pile):B=AF(j.leafPinnedAtPile吗,leaf=leaf,pile=pile);A=AW(B,listPermutationSpace);return A.get(K,[]),A.get(E,[])
def Co(state,leaf_k,leaf_r,domain_k=A,domain_r=A):
	D=leaf_k;C=domain_k;B=state
	if C is A:C=i(B,D)
	for E in reversed(N(C)):B=Bn(B,D,leaf_r,E,domainOf_leaf_r=domain_r)
	return B
def Bn(state,leaf_k,leaf_r,pile_k,domainOf_leaf_r=A):
	I=leaf_r;G=domainOf_leaf_r;F=leaf_k;C=pile_k;B=state;K=B.listPermutationSpace;B.listPermutationSpace=U();J=U();E=[]
	for D in K:
		if D.leafPinnedAtPile吗(F,C):E.append(D)
		elif Af(F,D.getLeafOptions(C,Ad(0))):H=D.copy();H[C]=Ak(H[C],F);B.listPermutationSpace.append(H);E.append(D.atPilePinLeaf(C,F))
		else:J.append(D)
	if G is A:G=i(B,I)
	for M in d(AN(0,C-L),G):E=Bo(E,I,M)
	B.listPermutationSpace.extend(E);B.reduceAllPermutationSpace(Bp).removeCreaseViolations();B.listPermutationSpace.extend(J);return B
def Bo(listPermutationSpace,leaf,pile):
	B=listPermutationSpace;A=pile;B,F=Bm(B,leaf,A);D=AW(AM('pilePinned吗',A),B);yield from D.get(E,[])
	for C in D.get(K,[]):C[A]=Ak(C[A],leaf);yield C
def A5(permutationSpace,pilesToUpdate,leafAntiOptions):
	B=permutationSpace
	for(D,E)in pilesToUpdate:
		C=Bj(Bk(leafAntiOptions,E))
		if C is A:B.clear()
		else:B[D]=C
	return B
def Ap(state,permutationSpace):
	A=permutationSpace;B=E
	while B:
		B=K;C,D=A.bifurcate()
		if not(A:=A5(A,h(D),A0(state.leavesTotal,Aa(C)))):return
		if T(C)<A.leafCount:B=E
	return A
def BB(state,permutationSpace):
	A=permutationSpace;B=E;G=0;H=1
	while B:
		B=K;I=A.leafCount;C=A.extractUndeterminedPiles();F={}
		for(J,D)in h(BR(B4(A_(C.values())),C)):F.setdefault(D,k()).add(J)
		for(D,L)in h(BQ(lambda groupBy:Bi(groupBy[G])==T(groupBy[H]),F)):
			if not(A:=A5(A,h(A7(B4(L),C)),A0(state.leavesTotal,AH(D)))):return
		if A.leafCount<I:B=E
	return A
def BC(state,permutationSpace):
	C=state;B=permutationSpace;D=E
	while D:
		D=K;H,I=B.bifurcate();J=Counter(B1(B1.from_iterable(x(AH,Aa(I))),Aa(H)))
		if k(G(C.leavesTotal)).difference(J.keys()):return
		L=k(AZ(BS((1).__eq__,J))).difference(H.values()).difference([C.leavesTotal])
		if L:
			M=L.pop();F=Ap(C,B.atPilePinLeaf(one(AZ(A8(Af(M),I))),M))
			if F is A or not F:return
			else:B=F
			D=E
	return B
Bp=Ap,BC,BB
def BD(state,maximumSizeListPermutationSpace,pileProcessingOrder,*,CPUlimit=A):
	C=pileProcessingOrder;A=state;H=Ac(CPUlimit)
	while C and T(A.listPermutationSpace)<maximumSizeListPermutationSpace:
		B=C.popleft();D=BY(AF(j.pileUndetermined吗,pile=B),A.listPermutationSpace);A.listPermutationSpace=U(D[K])
		with AV(H)as I:
			F=[I.submit(Bq,t(mapShape=A.mapShape,permutationSpace=C,pile=B))for C in D[E]]
			for G in AX(AU(F),total=T(F),desc=f"Pinning pile {B:3d} of {A.pileLast:3d}",disable=K):A.listPermutationSpace.extend(G.result().listPermutationSpace);A.listFolding.extend(G.result().listFolding)
	return A
def Bq(state):A=state;A.listPermutationSpace.extend(A.permutationSpace.deconstructAtPile(A.pile,d(A.pinAt_pile吗,Br(A))));return A.reduceAllPermutationSpace(Ag).removeCreaseViolations().moveToListFolding()
def Br(state):
	A=state;B=A6()
	if A.pile==Ae:B=A6([z])
	elif A.pile==C:B=A6([C])
	elif A.pile==Y(C)+A.首:B=A6([H(A.dimensionsTotal)])
	elif A.pile==D:B=B_(A)
	elif A.pile==Y(D)+A.首:B=C0(A)
	elif A.pile==D+C:B=C1(A)
	elif A.pile==Y(C+D)+A.首:B=C2(A)
	elif A.pile==J:B=C3(A)
	elif A.pile==Y(J)+A.首:B=C4(A)
	elif A.pile==Y(C)+H(A.dimensionsTotal):B=C5(A)
	return B
def AB(state,pileDepth=4,maximumSizeListPermutationSpace=2**14,*,CPUlimit=A):
	G=pileDepth;A=state
	if not u(A.mapShape):return A
	if not A.listPermutationSpace:A.listPermutationSpace.append(j().addMissingPileLeafSpace(B7(A)))
	E=V(BT((G,),'pileDepth',I),0)
	if E<0:H=f"I received `pileDepth = {G!r}`, but I need a value greater than or equal to 0.";raise ValueError(H)
	B=U()
	if 0<E:B.extend([Ae])
	if 1<=E:B.extend([C,Y(C)+A.首])
	if 2<=E:B.extend([D,Y(D)+A.首])
	if 3<=E:B.extend([D+C,Y(C+D)+A.首])
	if 4<=E:
		F=4
		if F<A.dimensionsTotal:B.extend([J])
		F=5
		if F<A.dimensionsTotal:B.extend([Y(J)+A.首])
	return BD(A,maximumSizeListPermutationSpace,B,CPUlimit=CPUlimit)
def Cp(state,maximumSizeListPermutationSpace=2**14,*,CPUlimit=A):
	B=maximumSizeListPermutationSpace;A=state
	if not u(A.mapShape):return A
	if not A.listPermutationSpace:A=AB(A,0)
	A=AB(A,4,B)
	if not u(A.mapShape,youMustBeDimensionsTallToPinThis=6):return A
	D=U([Y(C)+H(A.dimensionsTotal)]);return BD(A,B,D,CPUlimit=CPUlimit)
def AI(state,leaves,leavesDomain,*,youMustBeDimensionsTallToPinThis=3,CPUlimit=A):
	B=leaves;A=state
	if not u(A.mapShape,youMustBeDimensionsTallToPinThis=youMustBeDimensionsTallToPinThis):return A
	if not A.listPermutationSpace:A=AB(A,0)
	E=A.listPermutationSpace;A.listPermutationSpace=U()
	with AV(Ac(CPUlimit))as F:
		C=[F.submit(Bs,t(A.mapShape,permutationSpace=C),B,leavesDomain)for C in E]
		for D in AX(AU(C),total=T(C),desc=f"Pinning leaves {", ".join(x(f"{{:{T(str(A.leafLast))}d}}".format,B))} of {A.leafLast}",disable=K):A.listPermutationSpace.extend(D.result().listPermutationSpace);A.listFolding.extend(D.result().listFolding)
	return A
def Bs(state,leaves,leavesDomain):A=state;A.listPermutationSpace=A.permutationSpace.deconstructByDomainsCombined(leaves,leavesDomain);return A.reduceAllPermutationSpace(Ag).removeCreaseViolations().moveToListFolding()
def Bt(state,leaf,getLeafDomain,*,youMustBeDimensionsTallToPinThis=3,CPUlimit=A):
	B=leaf;A=state
	if not u(A.mapShape,youMustBeDimensionsTallToPinThis=youMustBeDimensionsTallToPinThis):return A
	if not A.listPermutationSpace:A=AB(A,0)
	E=Ac(CPUlimit);F=A.listPermutationSpace;A.listPermutationSpace=U()
	with AV(E)as G:
		C=[G.submit(Bu,state=t(mapShape=A.mapShape,permutationSpace=C),leaves=B,leavesDomain=getLeafDomain(t(mapShape=A.mapShape,permutationSpace=C),B))for C in F]
		for D in AX(AU(C),total=T(C),desc=f"Pinning leaf {B:16d} of {A.leafLast:3d}",disable=K):A.listPermutationSpace.extend(D.result().listPermutationSpace);A.listFolding.extend(D.result().listFolding)
	return A
def Bu(state,leaves,leavesDomain):A=state;A.listPermutationSpace=A.permutationSpace.deconstructByDomainOfLeaf(leaves,leavesDomain);return A.reduceAllPermutationSpace(Ag).removeCreaseViolations().moveToListFolding()
def Cq(state,*,CPUlimit=A):A=state;B=z,H(A.dimensionsTotal);return AI(A,B,leavesDomain=((Ae,A.pileLast),),CPUlimit=CPUlimit)
def Bv(state,*,CPUlimit=A):A=state;B=C+H(A.dimensionsTotal);return Bt(A,B,CW,CPUlimit=CPUlimit)
def Bw(state,*,CPUlimit=A):A=state;A=AB(A,0);return Bv(A,CPUlimit=CPUlimit)
def Bx(state,*,CPUlimit=A):A=state;B=D+C,D,Q(A.dimensionsTotal),e(A.dimensionsTotal);return AI(A,B,CK(A),CPUlimit=CPUlimit)
def By(state,*,CPUlimit=A):B=CPUlimit;A=state;A=Bx(A,CPUlimit=B);return Bw(A,CPUlimit=B)
def Cr(state,*,CPUlimit=A):A=state;B=J+D,J+D+C,J+C,J;return AI(A,B,CM(A),youMustBeDimensionsTallToPinThis=5,CPUlimit=CPUlimit)
def Bz(state,*,CPUlimit=A):A=state;B=m(A.dimensionsTotal),AR(A.dimensionsTotal),s(A.dimensionsTotal),A2(A.dimensionsTotal);return AI(A,B,CO(A),youMustBeDimensionsTallToPinThis=5,CPUlimit=CPUlimit)
def Cs(state,*,CPUlimit=A):A=state;return AI(A,(D+C,D),N((A,A+1)for A in i(A,D+C)),CPUlimit=CPUlimit)
def Ct(state,*,CPUlimit=A):A=state;return AI(A,(Q(A.dimensionsTotal),e(A.dimensionsTotal)),N((A,A+1)for A in i(A,Q(A.dimensionsTotal))),CPUlimit=CPUlimit)
def AJ(state,leaf):
	B=state;A=leaf
	if 0<A:return N(AC(B,abs(A)))
	return N(Aq(B,abs(A)))
def B_(state):
	A=state;B=sub;F=[];H=P(A.permutationSpace.getLeaf(B(A.pile,1)),f"I could not find an `int` type `Leaf` at {B(A.pile,1)}.");I=AJ(A,B(0,H));E=A.permutationSpace.getLeaf(Y(D)+A.首)
	if E and 0<S(E):F.extend([*G(S(E)-C,A.dimensionsTotal-D)])
	return p(I,F)
def C0(state):
	A=state;B=A9;F=[];H=P(A.permutationSpace.getLeaf(B(A.pile,1)),f"I could not find an `int` type `Leaf` at {B(A.pile,1)}.");I=AJ(A,B(0,H));E=A.permutationSpace.getLeaf(D)
	if E and E.bit_length()<A.dimensionsTotal:F.extend([*G(C,M(E)+L)])
	return p(I,F)
def C1(state):
	A=state;B=sub;E=[];J=P(A.permutationSpace.getLeaf(B(A.pile,1)),f"I could not find an `int` type `Leaf` at {B(A.pile,1)}.");F=AJ(A,B(0,J));K=P(A.permutationSpace.getLeaf(D));I=P(A.permutationSpace.getLeaf(Y(D)+A.首))
	if 1<T(F):E.append(0)
	if g(I)and K==C+H(A.dimensionsTotal):E.extend([*G(S(I)+C,A.dimensionsTotal)])
	return p(F,E)
def C2(state):
	A=state;B=A9;E=[];J=P(A.permutationSpace.getLeaf(B(A.pile,1)),f"I could not find an `int` type `Leaf` at {B(A.pile,1)}.");K=AJ(A,B(0,J));F=P(A.permutationSpace.getLeaf(D));I=P(A.permutationSpace.getLeaf(Y(D)+A.首))
	if I<e(A.dimensionsTotal):E.append(-1)
	if I==C+H(A.dimensionsTotal)and F!=D+C:E.extend([*G(M(F)-C)])
	return p(K,E)
def C3(state):
	B=state;N=sub;E=[];Q=P(B.permutationSpace.getLeaf(N(B.pile,1)),f"I could not find an `int` type `Leaf` at {N(B.pile,1)}.");R=AJ(B,N(0,Q));O=P(B.permutationSpace.getLeaf(D));L=P(B.permutationSpace.getLeaf(Y(D)+B.首));K=P(B.permutationSpace.getLeaf(D+C));J=P(B.permutationSpace.getLeaf(Y(C+D)+B.首))
	if b(K):E.extend([*G(M(K),5),Ai(K)]);E.append((F(v(L))+4)%5)
	if g(K):
		E.extend([*G(B.dimensionsTotal-3)][B.dimensionsTotal-3-(B.dimensionsTotal-2-v(J-(J.bit_count()-g(J))).bit_count())%(B.dimensionsTotal-2)-g(J):A])
		if g(L):E.extend([*G(S(v(L))-D,B.dimensionsTotal-3)])
	if O==C+H(B.dimensionsTotal):
		E.extend([(F(v(L))+4)%5,S(J)-1])
		if C+H(B.dimensionsTotal)<J:E.extend([*G(I(J-I(f(0,M(J)))).bit_length()-1,B.dimensionsTotal-2)])
		if 0<K-O<=f(0,B.dimensionsTotal-4)and 0<L-K<=f(0,B.dimensionsTotal-3):E.extend([Ai(K),B.dimensionsTotal-3,B.dimensionsTotal-4])
	return p(R,E)
def C4(state):
	B=state;M=A9;A=[];U=P(B.permutationSpace.getLeaf(M(B.pile,1)),f"I could not find an `int` type `Leaf` at {M(B.pile,1)}.");V=AJ(B,M(0,U));O=P(B.permutationSpace.getLeaf(D));W=P(B.permutationSpace.getLeaf(Y(D)+B.首));T=P(B.permutationSpace.getLeaf(D+C));N=P(B.permutationSpace.getLeaf(Y(C+D)+B.首));X=P(B.permutationSpace.getLeaf(J));K=N-W;E=X-T;H=T-O;I=O-C
	if E in{D,J,R,q}or E==五 and K!=D or H in{J,R}or H==D and not(I==K and E<0):
		if N==Q(B.dimensionsTotal):
			if I==R:A.append(F(J))
			if I==五:
				if H==J:A.append(F(J))
				if H==R:A.append(F(R))
			if E==R:A.append(F(J))
		if 0<(Z:=S(N))<5:A.extend(AE(G(Z%4))or[F(D)])
		if K==Y(五):A.append(F(D))
		if K==D:A.append(F(J))
		if K==q:
			if I==R:A.extend([*G(F(D),F(J)+L)])
			if H==D:
				if E==R:A.append(F(J))
		if I==D:
			A.append(F(D))
			if E==R:A.extend([*G(F(J),F(R)+L)])
			if E==q:A.extend([*G(F(R),F(q)+L)])
		if I==J:A.extend([*G(F(D),F(J)+L)])
		if I==R:A.append(F(R))
		if H==J:A.append(F(D))
		if H==R:A.extend([*G(F(D),F(J)+L)])
		if H==q:
			A.append(F(D))
			if E==R:A.extend([*G(F(D),F(R)+L)])
		if E==D:A.append(F(D))
		if E==J:A.append(F(J))
		if E==R:A.append(F(R))
		if E==五:A.append(F(D))
	return p(V,A)
def C5(state):
	A=state;h=P(A.permutationSpace.getLeaf(D));l=P(A.permutationSpace.getLeaf(Y(D)+A.首));Z=P(A.permutationSpace.getLeaf(D+C));U=P(A.permutationSpace.getLeaf(Y(C+D)+A.首));K=P(A.permutationSpace.getLeaf(J));L=P(A.permutationSpace.getLeaf(Y(J)+A.首));y=B7(A);E=[];x=D
	for(B,z)in A4(AH(y[x])):
		if z==h:
			if B<A.dimensionsTotal-2:E.extend([D,H(A.dimensionsTotal)+h])
			if 0<B<A.dimensionsTotal-2:E.extend([D+h])
			if B==1:E.extend([H(A.dimensionsTotal)+h+C])
			if B==A.dimensionsTotal-2:E.extend([Q(A.dimensionsTotal),Q(A.dimensionsTotal)+h])
	del x
	if h==C+H(A.dimensionsTotal):E.extend([Q(A.dimensionsTotal),l+C])
	if M(h)<A.dimensionsTotal-3:E.extend([D,l+D])
	x=Y(D)+A.首
	for(B,z)in A4(AH(y[x])):
		if z==l:
			if B==0:E.extend([D])
			if B<A.dimensionsTotal-2:E.extend([Q(A.dimensionsTotal)+l])
			if 0<B<A.dimensionsTotal-2:E.extend([V(A.productsOfDimensions,B),Q(A.dimensionsTotal)+l-V(A.sumsOfProductsOfDimensions,B)])
			if 0<B<A.dimensionsTotal-3:E.extend([C+l])
			if 0<B<A.dimensionsTotal-1:E.extend([Q(A.dimensionsTotal)])
	del x
	if h==C+m(A.dimensionsTotal)and l==e(A.dimensionsTotal):E.extend([m(A.dimensionsTotal),s(A.dimensionsTotal)])
	E.extend([Z])
	if Z==R+J+C:E.extend([J+D+C,C+J+H(A.dimensionsTotal)])
	if Z==C+J+Q(A.dimensionsTotal):E.extend([m(A.dimensionsTotal),Z+V(A.productsOfDimensions,P(a(Z))),Z+V(A.sumsOfProductsOfDimensions,P(a(Z))+1),s(A.dimensionsTotal)])
	if Z==C+A2(A.dimensionsTotal):E.extend([Q(A.dimensionsTotal)+(D+C),last(AC(A,v(Z)))])
	if Z==C+e(A.dimensionsTotal):E.extend([s(A.dimensionsTotal)])
	if b(Z):
		t=P(a(Z));AD=t*c+c;E.extend([V(A.productsOfDimensions,t)])
		if Z<H(A.dimensionsTotal):
			AE=AA(A.productsOfDimensions,A.dimensionsTotal,A.dimensionsTotal-1);E.extend([D,Z+V(A.sumsOfProductsOfDimensions,A.dimensionsTotal-1),Z+V(AE,AD)])
			if t==2:E.extend([V(A.sumsOfProductsOfDimensions,t)+V(A.productsOfDimensions,M(Z)),V(A.sumsOfProductsOfDimensions,t)+H(A.dimensionsTotal)])
			if t==3:E.extend([D+Z+V(A.productsOfDimensions,A.dimensionsTotal-1)])
		if H(A.dimensionsTotal)<Z:E.extend([C+e(A.dimensionsTotal),V(A.productsOfDimensions,M(Z)-1)])
	E.extend([U])
	if H(A.dimensionsTotal)<U:
		E.extend([C+e(A.dimensionsTotal)])
		if g(U):
			E.extend([Q(A.dimensionsTotal)]);B=D
			if O(U,F(B)):E.extend([B,H(A.dimensionsTotal)+B+C,A.首-A3(A.productsOfDimensions[F(B):A.dimensionsTotal-2]),U-B-V(A.sumsOfProductsOfDimensions,F(B)+1)])
			B=J
			if O(U,F(B)):
				E.extend([B,H(A.dimensionsTotal)+B+C])
				if 1<S(U):E.extend([A.首-A3(A.productsOfDimensions[F(B):A.dimensionsTotal-2])])
				else:E.extend([V(N(AC(A,v(U))),F(B))-C])
			B=R
			if O(U,F(B)):
				if 1<S(U):E.extend([B]);E.extend([A.首-A3(A.productsOfDimensions[F(B):A.dimensionsTotal-2])])
				if S(U)<F(B):E.extend([H(A.dimensionsTotal)+B+C])
			u=0;A0=I(f(0,A.dimensionsTotal-5))
			if U//A0&X(5)==21:
				E.extend([J]);u=Ai(U//A0)
				if 0<u<A.dimensionsTotal-3:o=A.productsOfDimensions[M(U)]-J;E.extend([U-o])
				if 0<u<A.dimensionsTotal-4:o=A.productsOfDimensions[P(a(U))]-J;E.extend([U-o])
		if b(U):
			E.extend([D])
			if U&X(4)==9:E.extend([11])
			u=Ai(U)
			if 0<u<A.dimensionsTotal-3:o=A.productsOfDimensions[M(U)]-D;E.extend([U-o])
			if 0<u<A.dimensionsTotal-4:o=A.productsOfDimensions[P(a(U))]-D;E.extend([U-o])
	if h==D+C and U!=next(AC(A,C+H(A.dimensionsTotal))):E.append(Q(A.dimensionsTotal))
	r=M(K);A1=N(Aq(A,K));w=[]
	if J<K<Y(C)+Q(A.dimensionsTotal):
		E.extend([K+H(A.dimensionsTotal)]);B=D
		if O(K,F(B)):E.extend([K+H(A.dimensionsTotal)+B])
		if not O(K,F(B)):E.extend([K+H(A.dimensionsTotal)-B])
		if b(K):
			B=R
			if O(K,F(B)):
				E.extend([K+H(A.dimensionsTotal)+B]);B=q
				if not O(K,F(B)):E.extend([K+H(A.dimensionsTotal)-B])
	if Q(A.dimensionsTotal)<K<H(A.dimensionsTotal)and P(a(K))!=2:
		E.extend([K+H(A.dimensionsTotal)])
		if b(K):
			B=J
			if not O(K,F(B)):E.extend([K+H(A.dimensionsTotal)-V(A.sumsOfProductsOfDimensions,F(B))])
			B=R
			if not O(K,F(B)):E.extend([K+H(A.dimensionsTotal)-B,K+H(A.dimensionsTotal)+V(A.sumsOfProductsOfDimensions,F(B))])
			B=q
			if O(K,F(B)):E.extend([K-B])
	if g(K):
		w.extend(G(A.dimensionsTotal-r+1,A.dimensionsTotal-AK));E.extend([K+C,K+H(A.dimensionsTotal),K+V(A.sumsOfProductsOfDimensions,A.dimensionsTotal-1),V(A.productsOfDimensions,r)+(D+C)]);B=D
		if O(K,F(B)):E.extend([B,H(A.dimensionsTotal)+B+C])
		B=J
		if not O(K,F(B)):w.append(A1.index(A.productsOfDimensions[r]))
		if K<H(A.dimensionsTotal):E.extend([V(A.productsOfDimensions,F(J)),V(A.sumsOfProductsOfDimensions,F(J)+1)])
		B=q
		if not O(K,F(B))and H(A.dimensionsTotal)<K:E.extend([V(A.productsOfDimensions,F(B))])
		A5=2
		if A.dimensionsTotal-AK-r==A5:AF=AA(A.productsOfDimensions,A.dimensionsTotal,A.dimensionsTotal-A5);AG=-1;AI=[A+AG for A in d(Ar,AF)];E.extend(AI)
	if b(K):
		if S(K-1)==1:E.extend([D])
		if v(K)==A.sumsOfProductsOfDimensions[3]:E.extend([J])
		B=C
		if O(K,F(B)):E.extend([B,K-B,H(A.dimensionsTotal)+B+C])
		B=J
		if not O(K,F(B)):w.append(F(B))
		if O(K,F(B))and O(K,F(D)):E.extend([K-B,H(A.dimensionsTotal)+B+C])
		B=R
		if O(K,F(B)):E.extend([K-B,H(A.dimensionsTotal)+B+C])
		if not O(K,F(B)):
			w.append(F(B));B=q
			if not O(K,F(B)):w.append(F(B))
		B=q
		if O(K,F(B)):
			i=C
			if O(K,F(i)):E.extend([H(A.dimensionsTotal)+B+i])
			i=J
			if O(K,F(i)):E.extend([H(A.dimensionsTotal)+B+i])
			i=R
			if O(K,F(i)):E.extend([H(A.dimensionsTotal)+B+i])
		B=五
		if O(K,F(B)):E.extend([Q(A.dimensionsTotal),C+e(A.dimensionsTotal)])
		if K<Q(A.dimensionsTotal):E.extend([D])
		if Q(A.dimensionsTotal)<K<H(A.dimensionsTotal):E.extend([K+V(A.sumsOfProductsOfDimensions,A.dimensionsTotal-2),Q(A.dimensionsTotal)+(D+C)])
		if H(A.dimensionsTotal)<K:
			B=J
			if O(K,F(B)):E.extend([K-B,H(A.dimensionsTotal)+B+C])
			B=q
			if O(K,F(B)):
				E.extend([B,K-B,H(A.dimensionsTotal)+B+C,s(A.dimensionsTotal)])
				if O(K,F(R)):E.extend([K-五])
	E.extend(p(A1,w));r=M(L);j=S(L)
	if O(V(y,Y(J)+A.首),L-1):
		B=R
		if not O(L,F(B)):
			AJ=AK
			for(A6,A7)in A4(N(AC(A,L-1)),start=AJ):
				if O(L,A6):E.extend([A7])
				if r<A6:E.extend([A7])
	A8=1
	if O(L,A8):
		A9=N(AC(A,L));AB=A.dimensionsTotal-1
		if T(A9)==AB:
			AL=2
			if not O(L,AL+A8):AM=A9[AB-AK];E.extend([AM])
	if L!=C+Q(A.dimensionsTotal):E.extend([C+e(A.dimensionsTotal)])
	if W(L)==1:E.extend([v(L)])
	B=J
	if O(L,F(B)):
		E.extend([L-B])
		if g(L)or b(L)and F(B)<CF(A,L):E.extend([B])
	B=R
	if O(L,F(B)):
		E.extend([L-B]);B=q
		if g(L)and not O(L,F(B)):E.extend([L-V(A.sumsOfProductsOfDimensions,F(B))])
	if j==3:E.extend([V(A.sumsOfProductsOfDimensionsNearest首,j)])
	if H(A.dimensionsTotal)<L:
		B=D
		if O(L,F(B)):E.extend([B,H(A.dimensionsTotal)+B+C])
		if b(L)and not O(L,F(B)):
			E.extend([L-H(A.dimensionsTotal)-B]);B=J
			if O(L,F(B)):E.extend([H(A.dimensionsTotal)+V(A.sumsOfProductsOfDimensions,F(B))])
		B=J
		if O(L,F(B)):
			E.extend([H(A.dimensionsTotal)+B+C]);B=R
			if g(L)and O(L,F(B)):E.extend([B])
		B=q
		if O(L,F(B)):E.extend([L-B])
		if not O(L,F(B)):E.extend([L+B])
	if b(L):
		B=C
		if O(L,F(B)):E.extend([D,L-B,L-V(A.productsOfDimensions,P(a(L)))])
	if g(L):
		B=C
		if not O(L,F(B)):E.extend([L+B,A.productsOfDimensions[j],L-A.productsOfDimensions[j]])
		B=J
		if O(L,F(B)):
			E.extend([B])
			if H(A.dimensionsTotal)<L<s(A.dimensionsTotal):
				E.extend([L+j])
				if j==2:AN=(A.首-L)//2;E.extend([AN+L])
			if L<H(A.dimensionsTotal):E.extend([L+A.sumsOfProductsOfDimensions[j],A.首-L])
		if L<H(A.dimensionsTotal):
			E.extend([Q(A.dimensionsTotal),L+A.productsOfDimensions[M(L)+1]]);B=R
			if not O(L,F(B)):E.extend([B,L+B,A.sumsOfProductsOfDimensionsNearest首[F(B)]])
		if L!=D+H(A.dimensionsTotal):E.extend([Q(A.dimensionsTotal)])
	del r,j;return n(k(AH(y[A.pile])).difference(k(E)))
def C6(state,permutationSpace):
	B=state;A=permutationSpace;F=E
	while F:
		F=K;J=A.leafCount
		for((L,C),(M,D))in BX(A.items()):
			if r(C)and A1(D):H=(M,D),;I=Aq(B,C)
			elif A1(C)and r(D):H=(L,C),;I=AC(B,D)
			else:continue
			if not(A:=A5(A,H,A0(B.leavesTotal,k(G(B.leavesTotal)).difference(I)))):return
		if A.leafCount<J:F=E
	return A
def C7(state,permutationSpace):
	B=state;A=permutationSpace
	if not u(B.mapShape,youMustBeDimensionsTallToPinThis=6):return A
	C=BH(B);D=E
	while D:
		D=K;H=A.leafCount
		for(F,G)in h(A7(BA(B.pileLast),A8(Ar,A8(C.__contains__,A.extractPinnedLeaves())))):
			if F in C[G]and not(A:=A5(A,h(AM(Av)(A7(AN(F+L,B.pileLast-L),A,factory=j))),A0(B.leavesTotal,C[G][F]))):return
		if A.leafCount<H:D=E
	return A
def C8(state,permutationSpace):
	J=state;F=permutationSpace;C=B0;D=B0;H=[];N=E;O=U()
	for M in G(J.dimensionsTotal):a=Al(Cf(M),Bb(1));V=AW(a,h(F.extractPinnedLeaves()));W=w(get(K,V,()));X=w(get(E,V,()));O.append(B2((M,),(X,),AL(W.items(),2)));O.append(B2((M,),(W,),AL(X.items(),2)))
	while N:
		N=K;b=F.leafCount
		for(M,Y,((A,c),(B,d)))in concat(O):
			Q=I(f(c,M));R=I(f(d,M))
			if(S:=B9(Y,Q)):C=P(Ab(F,Q))
			if(T:=B9(Y,R)):D=P(Ab(F,R))
			if S and not T:
				Z=A0(J.leavesTotal,(R,))
				if A<B<C:H=A6([*G(A),*G(C+1,J.pileLast+L)])
				elif C<B<A:H=A6([*G(C),*G(A+1,J.pileLast+L)])
				elif B<C<A or C<A<B:H=G(C+1,A)
				elif B<A<C or A<C<B:H=G(A+1,C)
			elif not S and T:
				Z=A0(J.leavesTotal,(Q,))
				if D<A<B:H=A6([*G(D),*G(B+1,J.pileLast+L)])
				elif B<A<D:H=A6([*G(B),*G(D+1,J.pileLast+L)])
				elif A<B<D or B<D<A:H=G(B+1,D)
				elif A<D<B or D<B<A:H=G(D+1,B)
			elif S and T:
				if Aj(A,B,C,D):return
				continue
			else:continue
			if not(F:=A5(F,h(A7(Be(H),F.extractUndeterminedPiles())),Z)):return
		if b<F.leafCount:N=E
	return F
def C9(state,permutationSpace):
	B=state;A=permutationSpace;C=E
	while C:
		C=K;J=A.leafCount;N=2
		for(F,H)in h(A7(BA(B.pileLast),A8(Ar,A.extractPinnedLeaves()))):
			D=M(H)
			if 0<D and not(A:=A5(A,h(AM(Av)(A7(AN(N,F-L),A,factory=j))),A0(B.leavesTotal,G(B.productsOfDimensions[D],B.leavesTotal,B.productsOfDimensions[D])))):return
			I=S(H)
			if 0<I and not(A:=A5(A,h(AM(Av)(A7(AN(F+L,B.pileLast-L),A,factory=j))),A0(B.leavesTotal,G(z,B.sumsOfProductsOfDimensions[I])))):return
		if A.leafCount<J:C=E
	return A
def CA(state,permutationSpace):
	I=state;B=permutationSpace;G=E
	while G:
		G=K;J=B.leafCount
		for((L,C),(M,A),(N,D))in BZ(n(h(B))):
			if r(C)and r(A)and A1(D):H=(N,D),;F=A+(A-C)
			elif r(C)and A1(A)and r(D):H=(M,A),;F=(C+D)//2
			elif A1(C)and r(A)and r(D):H=(L,C),;F=A-(D-A)
			else:continue
			if 0<=F<I.leavesTotal and not(B:=A5(B,H,A0(I.leavesTotal,[F]))):return
		if B.leafCount<J:G=E
	return B
Ag=Ap,C6,BC,BB,C9,C7,C8,CA
Z=2
AQ=0
C=Z**AQ
CB=Z
AQ+=1
CC=AQ
D=CB**CC
CD=Z
AQ+=1
CE=AQ
J=CD**CE
R=Z**3
q=Z**4
五=Z**5
六=Z**6
七=Z**7
八=Z**8
九=Z**9
@B
def F(A,*,dimensionLength=Z):return I(log(A,dimensionLength))
@B
def H(A):return I('1'+o*(A-1),Z)
@B
def e(A):return I('11'+o*(A-2),Z)
@B
def s(A):return I('111'+o*(A-3),Z)
@B
def AR(A):return I('101'+o*(A-3),Z)
@B
def Q(A):return I('01'+o*(A-2),Z)
@B
def A2(A):return I('011'+o*(A-3),Z)
@B
def m(A):return I('001'+o*(A-3),Z)
@B
def Ah(A):return I('0001'+o*(A-4),Z)
@B
def Cu(A):return I('1111'+o*(A-4),Z)
@B
def Cv(A):return I('1101'+o*(A-4),Z)
@B
def Cw(A):return I('1011'+o*(A-4),Z)
@B
def Cx(A):return I('1001'+o*(A-4),Z)
@B
def Cy(A):return I('0111'+o*(A-4),Z)
@B
def Cz(A):return I('0101'+o*(A-4),Z)
@B
def C_(A):return I('0011'+o*(A-4),Z)
def u(mapShape,*,youMustBeDimensionsTallToPinThis=3):A=mapShape;return youMustBeDimensionsTallToPinThis<=T(A)and all(x((2).__eq__,A))
def CF(state,integerNonnegative):return Ax(CH(state.dimensionsTotal,integerNonnegative))or 0
@B
def M(A):return max(0,A.bit_length()-1)
@B
def a(B):
	C=I(f(B,M(B)))
	if C==0:D=A
	else:D=M(C)
	return D
@B
def AS(B):
	F=M(B);D=a(B)
	if D in{0,A}:C=A
	else:
		E=I(f(B,F).bit_flip(P(D)))
		if E==0:C=A
		else:C=M(E)
	return C
@B
def CG(B):
	F=M(B);G=a(B);D=AS(B)
	if D in{0,A}:C=A
	else:
		E=I(f(B,F).bit_flip(P(G)).bit_flip(P(D)))
		if E==0:C=A
		else:C=M(E)
	return C
@B
def v(A):return I(BN(A,M(A)))
@B
def S(A):return Ax(A)or 0
@B
def W(A):return max(0,A.bit_count()-1)
@B
def CH(dimensionsTotal,integerNonnegative):return I(integerNonnegative^X(dimensionsTotal))
@B
def Ai(A):return v(A-(D+C)).bit_count()
def AC(state,leaf):return iter(BE(state,leaf,increase=K))
def Aq(state,leaf):return iter(BE(state,leaf,increase=E))
def BE(state,leaf,*,increase=E):return CI(leaf,state.dimensionsTotal)[increase]
@B
def CI(leaf,dimensionsTotal):
	B=leaf;H=[I(f(B,A))for A in G(dimensionsTotal)]
	if B==z:J=[1];F=[]
	else:
		E=b(W(B));C=AT(E,M(B)*f(E,0)or A);D=AT(f(E,0),M(B)*E or A)
		if g(B):
			if C.start==1:C=AT(C.start+S(B),C.stop)
			if D.start==1:D=AT(D.start+S(B),D.stop)
		F=H[C];J=H[D]
		if B==1:F=[0]
	return N(F),N(J)
@B
def CJ(leaf,dimensionsTotal,mapShape,leavesTotal):
	A=leaf;B=t(mapShape)
	if u(B.mapShape):D=A==z;return G(B.sumsOfProductsOfDimensions[S(A)+L]+W(A)-D,B.sumsOfProductsOfDimensionsNearest首[M(A)]+2-W(A)-D,2+2*(A==H(dimensionsTotal)+C))
	return G(leavesTotal)
def CK(state):A=state;B=N(i(A,D+C));E=N(i(A,Q(A.dimensionsTotal)));return CL(B,E,A.dimensionsTotal)
@B
def CL(domain一零,domain首一,dimensionsTotal):
	A=dimensionsTotal;I=[]
	for B in domain一零:
		F=domain首一;C=T(F);E=[]
		if B<=m(A):0
		elif m(A)<B<Q(A):E.extend([*G(1,C//2),*G(1+C//2,3*C//4)])
		elif B==Q(A):E.extend([*G(1,C//2)])
		elif Q(A)<B<H(A)-D:E.extend([*G(3*C//4)])
		elif B==H(A)-D:E.extend([*G(1,3*C//4)])
		elif B==H(A):E.extend([*G(2,C//2)])
		F=N(p(F,E));I.extend([(B,B+1,A,A+1)for A in F])
	return N(d(Ao,I))
def CM(state):A=state;B=CQ(A);C=CR(A);return CN(B,C,A.dimensionsTotal)
@B
def CN(domain二零and二,domain二一零and二一,dimensionsTotal):
	b=domain二一零and二一;O=dimensionsTotal;E=N(d(AY,domain二零and二));X=N(d(AY,b));U=T(X);Y=[];c=N(I(f(0,A))for A in G(O+1))
	for(V,(F,A))in A4(X):
		H=[];D=S(A);e=V;H.extend(G(e));J=U
		if A<=Q(O):
			if D==1:
				J=U//2+V
				if W(A)==2:J-=1
				if W(A)==1 and 2<M(A):J+=2
				if W(A)==1 and M(A)-P(a(A))<2:C=c[O-2]+4;J=E.index((A+C,F+C))
			else:
				J=3*U//4+2
				if V==0:J=1
				elif V<=2:C=R+A3(c[1:O-2]);J=E.index((A+C,F+C))
		H.extend(G(J,U))
		if A<A2(O):
			if D==4:C=I(f(0,D));B=E.index((A+C,F+C));H.extend([*G(B,B+D)])
			if D==3:C=I(f(0,D));B=E.index((A+C,F+C));H.extend([*G(B,B+D-1)]);B=E.index((A+C*2,F+C*2));H.extend([*G(B-1,B+D-1)])
			if D<3 and 2<M(A):
				if 5<O:
					C=q;B=E.index((A+C,F+C));K=B+C;Z=2
					if D==1 and M(A)==4:B+=2;K=B+1
					if D==2:
						B+=3
						if M(A)==4:B-=2
						K=B+D+L
					if W(A)==2:K=B+1
					H.extend([*G(B,K,Z)])
				if M(A)==3 and W(A)==1 or M(A)-P(a(A))==3:
					C=A;B=E.index((A+C,F+C));K=B+2
					if D==2:B+=1;K+=1
					if M(A)==4:B+=3;K+=4
					Z=1;H.extend([*G(B,K,Z)])
			if M(A)==2:C=R;B=E.index((A+C,F+C));H.extend([*G(B,B+C,2)])
		Y.extend([(A,F,B,C)for(B,C)in p(E,H)])
	g=N(k(b).difference(k(X)));Y.extend([(B,A,A-1,B+1)for(A,B)in g]);return N(n(d(Ao,k(Y))))
def CO(state):A=state;B=CS(A);C=CU(A);return CP(A.dimensionsTotal,B,C)
@B
def CP(dimensionsTotal,domain首零二and首二,domain首零一二and首一二):
	L=domain首零二and首二;A=dimensionsTotal;O=N(d(AY,L));Q=N(d(AY,domain首零一二and首一二));E=T(Q);K=[]
	for(R,(B,C))in A4(O):
		F=[];J=S(B);U=R-1;F.extend(G(U));D=E
		if J==1:
			D=E-(I(C^X(A))//4-1)
			if W(C)==3 and A-M(C)>=2:D+=2
			if W(C)==1 and A-M(C)>=2 and M(C)-P(a(C))>3:D+=2
			if W(C)==1 and M(C)-P(a(C))>4:D+=2
			if W(C)==A-M(C)and 4<=M(C)and W(C)>1:D-=1
		else:
			if AR(A)<=B:D=E-1
			if H(A)<B<AR(A):D=E-(I(B^X(A))//8-1)
			if A2(A)<B<=H(A):D=E-I(X(A-4))
			if B==A2(A):D=E-I(X(A-4))-1
			if B<A2(A):D=E-I(X(A-3))-(J==2)
		F.extend(G(D,E))
		if J==1 and abs(B-H(A))==2 and g(A):F.extend([D-2])
		if J!=1 and A2(A)<=B<=e(A):
			if J==2 and W(B)+1!=M(B)-P(a(B)):
				F.extend([E-(I(B^X(A))//8+2)])
				if B<=H(A)and g(A):F.extend([E-(I(B^X(A))//4-1)])
			if J==3:F.extend([D-2])
			if 3<J:F.extend([E-I(B^X(A))//4])
		K.extend([(C,B,A,D)for(A,D)in p(Q,F)])
	V=N(k(L).difference(k(O)));K.extend([(B,A,A-1,B+1)for(A,B)in V]);return N(n(d(Ao,k(K))))
def CQ(state):A=state;B=N(i(A,J+C));D=N(i(A,J));E=A9;return BF(B,D,E,A.dimensionsTotal,A.sumsOfProductsOfDimensions)
def CR(state):A=state;B=N(i(A,J+D+C));E=N(i(A,J+D));F=sub;return BF(B,E,F,A.dimensionsTotal,A.sumsOfProductsOfDimensions)
@B
def BF(domain零,domain0,direction,dimensionsTotal,sumsOfProductsOfDimensions):
	c=direction;Z=domain零;U=domain0;L=dimensionsTotal
	if c(0,6009)==6009:e=E;h=K
	else:e=K;h=E
	i=[];P=T(Z);R=P-T(U)
	for(V,B)in A4(d(AN(Ae,H(L)-C),Z)):
		O=[];W=S(B-b(B));j=0;l=V
		if e:j=0;l=I(X(W-1).bit_flip(0))
		elif h:j=I(g(V)or W);l=V
		if e:
			if B==J:O.extend([*G(V+1)])
			if B==Q(L)+m(L)+Ah(L):A=I(7*P/8);A-=R;O.extend([A])
		o=V+j;o-=R;O.extend(G(o))
		if B<=Q(L):Y=V+3*P//4;Y-=R;O.extend(G(Y,P))
		if Q(L)<B<H(L):Y=I(B^X(L))//2;O.extend(G(Y,P))
		for q in G(W):O.extend(G(l+I(X(q)),P,I(f(0,q+1))))
		if W==1:
			if m(L)<B<H(L)-C and 2<M(B):
				if a(B)==C:
					A=P//2;A-=R
					if 4<U[A].bit_length():O.extend([A])
					if Q(L)<B:A=-(P//4-b(B));A-=-R;O.extend([A])
				if a(B)==D:
					A=P//2+2;A-=R
					if U[A]<H(L):O.extend([A])
					A=-(P//4-2);A-=-R
					if Q(L)<B:O.extend([A])
				if a(B)==D+C:A=-(P//4);A-=-R;O.extend([A])
				A=3*P//4;A-=R
				if B<A2(L):
					s=L;t=F(D);u=F(J);v=s-(t+u);r=sumsOfProductsOfDimensions[v]
					if h:r-=1
					w=r+H(L);A=U.index(w);O.extend([A])
				if AS(B)==C:
					if a(B)==D+C:O.extend([A-2])
					if M(B)==D+C:O.extend([A-2])
		elif Q(L)+Ah(L)+b(B)==B:A=3*P//4-1;A-=R;O.extend([A])
		i.extend([(B,A)for A in p(U,O)])
	i.extend([(A,c(A,C))for A in Z if c(A,C)in U]);return N(n(k(i)))
def CS(state):A=state;B=N(i(A,AR(A.dimensionsTotal)));C=N(i(A,m(A.dimensionsTotal)));return CT(B,C,A.dimensionsTotal)
@B
def CT(domain首零二,domain首二,dimensionsTotal):
	K=dimensionsTotal;Q=[];R=domain首零二;U=domain首二;O=sub;Q.extend([(A,O(A,C))for A in R if O(A,C)in U]);F=T(R);L=F-T(U)
	for(Y,A)in A4(R):
		if A<H(K)+C:continue
		E=[];P=S(O(A,b(A)))
		if e(K)<A:V=Y+3-3*F//4
		else:V=2+(e(K)-O(A,b(A)))//2
		V-=L;E.extend(G(V));Z=Y+2-I(X(P));Z-=L;E.extend(G(Z,F));d=F-1;d-=L;h=d-I(X(P-1).bit_flip(0))
		for g in G(P):E.extend(G(h-I(X(g)),c,c*I(f(0,g+1))))
		if P==1:
			if AS(A)==D and J+C<=M(A):
				B=F//2+1;B-=L;E.extend([B]);B=F//4+1;B-=L;E.extend([B])
				if A<e(K):E.extend([B-2])
			if W(A)==D:
				B=F//4+3;B-=L
				if a(A)==D:E.extend([B])
				if a(A)==J:E.extend([B])
				if M(A)==K-1 and a(A)==K-3 or a(A)==J:E.extend([B-2]);B=F//2-1;B-=L;E.extend([B])
		elif e(K)-O(Ah(K),b(A))==A:B=F//4+2;B-=L;E.extend([B])
		Q.extend([(A,B)for B in p(U,E)])
	return N(n(k(Q)))
def CU(state):A=state;B=N(i(A,s(A.dimensionsTotal)));C=N(i(A,A2(A.dimensionsTotal)));D=A9;return CV(B,C,D,A.dimensionsTotal)
@B
def CV(domain零,domain0,direction,dimensionsTotal):
	Q=domain零;P=domain0;O=direction;K=dimensionsTotal;R=[];F=T(Q);L=F-T(P)
	for(U,B)in A4(Q):
		if B<H(K):continue
		E=[];V=S(O(B,b(B)))
		if e(K)<B:Y=U+1-3*F//4
		else:Y=(e(K)-O(B,b(B)))//2
		Y-=L;E.extend(G(Y));Z=U+1-I(X(V));Z-=L;E.extend(G(Z,F));g=U
		for d in G(V):E.extend(G(g-I(X(d)),c,c*I(f(0,d+1))))
		if V==1:
			if AS(B)==D and J+C<=M(B):
				A=F//2;A-=L;E.extend([A]);A=F//4;A-=L;E.extend([A])
				if B<e(K):E.extend([A-2])
			if AS(B)==D+C:
				A=F//4;A-=L
				if CG(B)==D:E.extend([A])
			if W(B)==D:
				A=F//4+2;A-=L
				if a(B)==D:A=P.index(H(K)-D);E.extend([A])
				if a(B)==J:E.extend([A])
				if AR(K)<B and J+C<=M(B):E.extend([A-2]);A=F//2-2;A-=L;E.extend([A])
		elif e(K)-O(Ah(K),b(B))==B:A=F//4+1;A-=L;E.extend([A])
		R.extend([(B,A)for A in p(P,E)])
	R.extend([(A,O(A,C))for A in Q if O(A,C)in P]);return N(n(k(R)))
def CW(state,leaf=A):
	E=leaf;B=state
	if E is A:E=C+H(B.dimensionsTotal)
	F=N(i(B,E));G=D+C;I=e(B.dimensionsTotal)
	if B.permutationSpace.leafPinned吗(G)and B.permutationSpace.leafPinned吗(I):J=P(Ab(B.permutationSpace,G));K=P(Ab(B.permutationSpace,I));F=CX(F,J,K,B.dimensionsTotal,B.leavesTotal)
	return F
@B
def CX(domain首零Plus零,pileOfLeaf一零,pileOfLeaf首零一,dimensionsTotal,leavesTotal):
	P=leavesTotal;K=pileOfLeaf首零一;E=pileOfLeaf一零;A=dimensionsTotal;M=Q(A);R=1-I(E.bit_count()==1);S=A-(E.bit_length()+R);T=I(X(S));O=M-T;B=[]
	if E==J:B.extend([C,D,J])
	if J<E<=m(A):
		F=M//2-1;B.extend(G(1,F));U=5
		for Y in B3(A-U):V=1+F;F+=(F+1)//2;B.extend([*G(V,F)])
		B.extend([*G(1+F,O)])
	if m(A)<E:B.extend([*G(1,O)])
	R=1-I((P-K).bit_count()==1);S=A-((P-K).bit_length()+R);T=I(X(S));O=M-T;U=5
	if K==P-J:
		B.extend([-C-1,-D-1])
		if U<=A:B.extend([-J-1])
	if s(A)<K<P-J and m(A)<E<=H(A):B.extend([-1])
	if s(A)<=K<P-J:
		F=M//2-1;B.extend(G((1+L)*c,(F+L)*c,c))
		for Y in B3(A-U):V=1+F;F+=(F+1)//2;B.extend([*G((V+L)*c,(F+L)*c,c)])
		B.extend([*G((1+F+L)*c,(O+L)*c,c)])
		if J<=E<=H(A):B.extend([C,D,J,M//2])
	if K==s(A)and Q(A)<E<=H(A):B.extend([-1])
	if e(A)<K<s(A):
		if E in{Q(A),H(A)}:B.extend([-1])
		elif J<E<m(A):B.extend([0])
	if K<s(A):B.extend([*G((1+L)*c,(O+L)*c,c)])
	W=Q(A);R=1-I(W.bit_count()==1);S=A-(W.bit_length()+R);T=I(X(S));O=M-T
	if K==P-J:
		if E==J:B.extend([C,D,J,M//2-1,M//2])
		if J<E<=H(A):Z=O-1;B.extend([*G(1,3*M//4),*G(1+3*M//4,Z)])
		if Q(A)<E<=H(A):B.extend([-1])
	if K==e(A):
		if E==H(A):B.extend([-1])
		elif J<E<m(A)or m(A)<E<Q(A):B.extend([0])
	return N(p(domain首零Plus零,B))
def BG(state):A=state;return{B:i(A,B)for B in G(A.leavesTotal)}
def BH(state):
	A=state;B={}
	if u(A.mapShape,youMustBeDimensionsTallToPinThis=6):B=CY(A.mapShape)
	return B
@B
def CY(mapShape):
	B=t(mapShape);U=BG(B);O={}
	for V in G(3,B.dimensionsTotal+L):
		for f in G(V-2+c,c,c):
			for E in G(B.productsOfDimensions[V]-A3(B.productsOfDimensions[f:V-2]),B.leavesTotal,B.productsOfDimensions[V-1]):O[E]={A:[B.productsOfDimensions[M(E)]+B.productsOfDimensions[S(E)]]for A in AE(U[E])[0:AA(B.productsOfDimensions,dimensionFrom首=V-1)[V-2-f]//2]}
	E=C+Q(B.dimensionsTotal);O[E]={A:[2*B.productsOfDimensions[M(E)]+B.productsOfDimensions[S(E)],3*B.productsOfDimensions[M(E)]+B.productsOfDimensions[S(E)]]for A in AE(U[E])[1:2]};del E;E=C+e(B.dimensionsTotal);F=AE(U[E]);O[E]={A:[]for A in AE(U[E])};X=AA(B.productsOfDimensions);h=AA(B.productsOfDimensions,dimensionFrom首=B.dimensionsTotal-1);T=2
	for P in F[F.index(D+C):F.index(Y(C)+H(B.dimensionsTotal))+L]:O[E][P].append(C+H(B.dimensionsTotal))
	for K in G(B.dimensionsTotal-2):
		Z=B.sumsOfProductsOfDimensions[K+2];d=B.productsOfDimensions[W(Z)]
		for a in G(d):
			J=Z+a*c;R=X[K]+B.sumsOfProductsOfDimensions[2]+B.productsOfDimensions[B.dimensionsTotal-(K+2)]-T*2*(W(J)-1+g(J))*(1+(2==W(J)+g(J)==M(J)))
			for P in F[F.index(R):A]:O[E][P].append(J)
			N=J+H(B.dimensionsTotal)
			if v(J)==0 and b(S(J)):O[E][R].append(N)
			if N==E:continue
			R=F[-1]-T*(W(N)-1+g(N)-b(N)-I(S(N)==B.dimensionsTotal-2)-I(E<N))
			for P in F[F.index(R):A]:O[E][P].append(N)
			if K<B.dimensionsTotal-4 and b(S(J-b(J))):
				R=h[K]+B.sumsOfProductsOfDimensions[3+K]-T*2*(W(N)-1+g(N)*K-g(N)*I(not bool(K)))+B.productsOfDimensions[B.dimensionsTotal-1+a*I(not bool(K))-(K+2)]
				for P in F[F.index(R)+K:F.index(Y(C)+H(B.dimensionsTotal))-K+L]:O[E][P].append(N)
	del E,F,X,T,h;E=C+H(B.dimensionsTotal);F=AE(U[E])[1:A];O[E]={A:[]for A in F};X=AA(B.productsOfDimensions);T=4
	for K in G(B.dimensionsTotal-2):
		Z=B.sumsOfProductsOfDimensions[K+2];d=B.productsOfDimensions[W(Z)]
		for a in G(d):
			J=Z+a*c;N=J+H(B.dimensionsTotal);R=X[K]+6-T*(W(J)-1+g(J))
			for P in F[F.index(R):A]:O[E][P].append(J);O[E][P].append(N)
	del E,F,X,T
	if B.dimensionsTotal==6:
		E=22;i=AT(0,A);F=AE(U[E])[i];j=[(15,43,43)]
		for(J,R,k)in j:
			for l in F[F.index(R):F.index(k)+L]:O[E].setdefault(l,[]).append(J)
	return O
def D0(state):return CZ(state.mapShape)
@B
def CZ(mapShape):
	C=t(mapShape);D=BG(C);E={};I=BH(C)
	for(B,J)in I.items():
		K=N(D[B]);F=BM(k)
		for(L,M)in J.items():
			for A in M:F[A].add(L)
		for(A,O)in F.items():
			P=N(D[A]);Q=n(A for A in K if A not in O)
			for G in P:
				R=BL(Q,G)
				if R==0:
					H=E.setdefault(A,{}).setdefault(G,[])
					if B not in H:H.append(B)
	return E
@y
def Ca(pile,dimensionsTotal,leaf):B=dimensionsTotal;A=leaf;return pile<I(X(B)^X(B-M(A)))-W(A)+2-(A==z)
@y
def Cb(pile,leaf):A=leaf;return I(f(0,S(A)+1))+W(A)-1-(A==z)<=pile
@y
def Cc(pile,leaf):A=leaf;return pile&1==I(f(0,S(A)+1))+W(A)-1-(A==z)&1
@y
def Cd(pile,dimensionsTotal,leaf):
	A=leaf
	if A!=H(dimensionsTotal)+C:return E
	return pile>>1&1==I(f(0,S(A)+1))+W(A)-1-(A==z)>>1&1
@B
def Ce(pile,dimensionsTotal,mapShape,leavesTotal):
	D=leavesTotal;C=dimensionsTotal;B=pile;A=G(D)
	if u(mapShape):E=Cc(B);F=Cb(B);H=Ca(B,C);I=Cd(B,C);A=d(E,A);A=d(F,A);A=d(H,A);A=d(I,A)
	return Bh(D,A)
def Ar(leaf):return C<leaf
@y
def Cf(dimension,leaf):return O(leaf,dimension)
def Aj(pile,pileComparand,pileCrease,pileComparandCrease):
	D=pile;C=pileComparandCrease;B=pileComparand;A=pileCrease
	if D<B:
		if C<D:
			if A<C:return E
			return B<A
		if B<A:return A<C
		else:return D<C<A<B
	return K
def D1(folding,mapShape):
	A=mapShape;B=w(A4(folding));I={B:A for(A,B)in h(B)}
	for C in G(BI(A)):
		D=[U(),U()]
		for(J,F)in B.items():
			H=As(A,F,C)
			if H:D[At(A,F,C)].append((J,I[H]))
		for L in D:
			if any(Aj(A,C,B,D)for((A,B),(C,D))in AL(n(L),2)):return K
	return E
def D2(leavesPinned,mapShape):
	B=leavesPinned;A=mapShape;I={B:A for(A,B)in h(B)}
	for C in G(BI(A)):
		D=[U(),U()]
		for(J,F)in B.items():
			H=As(A,F,C)
			if H:D[At(A,F,C)].append((J,I[H]))
		for L in D:
			if any(Aj(A,C,B,D)for((A,B),(C,D))in AL(n(L),2)):return K
	return E
@B
def BI(mapShape):return T(mapShape)
@B
def D3(mapShape):return B5(mapShape)
@B
def As(mapShape,leaf,dimension):
	C=dimension;B=mapShape;D=A
	if leaf//Au(B,C)%B[C]+1<B[C]:D=leaf+Au(B,C)
	return D
@B
def At(mapShape,leaf,dimension):B=dimension;A=mapShape;return leaf//Au(A,B)%A[B]&1
@B
def Au(mapShape,dimension):return prod(mapShape[0:dimension],start=1)
def Cg(state):
	A=state;C=U()
	while A.listPermutationSpace:D=A.listPermutationSpace.pop();B=t(A.mapShape,permutationSpace=D);B.listPermutationSpace.extend(B.permutationSpace.deconstructAtPile());B=B.reduceAllPermutationSpace(Ag).removeCreaseViolations().moveToListFolding();C.extend(B.listFolding);A.listPermutationSpace.extend(B.listPermutationSpace)
	A.listFolding.extend(C);return A
def Ch(state,workersMaximum):
	A=state
	if not u(A.mapShape):return A
	if not A.listPermutationSpace:A=AB(A,1)
	with AV(workersMaximum)as C:
		D=A.listPermutationSpace.copy();A.listPermutationSpace=U();B=[C.submit(Cg,t(A.mapShape,listPermutationSpace=U([B])))for B in D]
		for E in AX(AU(B),total=T(B),disable=K):A.listFolding.extend(E.result().listFolding)
	A.Theorem4Multiplier=BV(A.dimensionsTotal);A.groupsOfFolds=T(A.listFolding);return A
if __name__=='__main__':Ci=A;AD=t((2,)*5);AD=AB(AD,4);AD=Bz(AD);AD=By(AD);Cj=Ac(Ci);print(Ch(AD,Cj).foldsTotal)