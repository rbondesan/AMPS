"""Class SU2kMPS:

Created by RB on 25/01/17. Based on AbelianTensor class of 1512-03846
and Phys Rev B 89, 075112 (2014).

"""

import numpy as np
import collections
import itertools
import heapq
import warnings
from functools import reduce
import operator
from copy import deepcopy
from numpy.linalg import inv as np_inv

# auxliary files
from modules import SU2k_data

def generate_binary_deferer(op_func):
    def deferer(self, B, *args, **kwargs):
        return type(self).defer_binary_elementwise(self, B, op_func, *args,
                                                   **kwargs)
    return deferer

def generate_unary_deferer(op_func):
    def deferer(self, *args, **kwargs):
        return type(self).defer_unary_elementwise(self, op_func, *args,
                                                  **kwargs)
    return deferer

class SU2kMPS:
    """A class for MPS whose legs carry spaces = \oplus_h n_h V_h, where
    V_h is a one dimensional space associated to an SU2k anyon of type
    (height, irrep) h and n_h is its multiplicity.  The MPS ecnodes a
    compressed fusion tree, where compressed refers to the fact that
    the degeneracy index t is associated to paths on a Brattelli diag
    which end at h. In notation of Phys Rev B 89, 075112 (2014),
    mu=(h,t).  The SU2kMPS encodes these space in the following order:
    
       |   |   |
       1   3   5   
    _0_|_2_|_4_|_6_ ...


    Note: the internal horizontal legs carry physical indices, the
    heights, and the auxiliary, multiplicity, indices are summed
    over. Therefore, their shape (see below) is [1], since already
    contracted.

    Every SU2kMPS has the following attributes:

    shape: A list of dims, one dim per leg. Every dim is a list of
    integers that are the multiplicity along that leg.

    heights: A list of heights, each corresponding to an
    anyon type on a leg: h = 2j+1 which are integers: 1,2,3,4,...,hmax
    The heights (or quantum numbers) are in one-to-one correspondence with the
    elements of the dims, so that heights[i][j] and shape[i][j] are the
    qnum and dimension of the same block. 

    REMOVED (Our anyons are self-dual)
    dirs: A list of integers -1 or 1, one for each leg. 1 means 
    that the corresponding leg is outgoing, -1 means incoming.

    hmax: k + 1 in SU(2)k, which truncates the fusion rules.
    It is the maximum height. If None, then k=infty and one is back to su(2).

    sects: A dict of numpy arrays, with combinations of qnums
    as keys. Every key must a tuple of anyon types, one for each
    leg, and each one of them being from the qim of that leg. The value
    of the dict at this key is the block (or "sector" or "sect")
    corresponding to these quantum numbers. If the tensor is invariant
    (see invar), then only certain blocks are allowed to
    be set, but even in such a cause not all allowed blocks must be set. 
    For the treatement of unset blocks see defval.
    
    dtype: A numpy dtype, that is the dtype of all the sects.

    defval: The default value that the tensor has everywhere outside the
    blocks set in sects. If the tensor is a scalar with no legs then its 
    value is its defval and it has no blocks. Note that many of the
    methods - such as dot and svd - require defval == 0 (and assert
    this). The main use of defval != 0 is to be able to handle tensors
    of boolean values that arise in comparisons.                                                   

    charge: An integer such that if invar is True then all the blocks
    set in sects must have keys k such that the corresponding
    anyon types fuse into charge.                                                           

    invar: A boolean. If True, then the tensor is invariant in the sense
    explained in the definition charge. If False, this condition is ignored 
    and any block can be set. Note that as with defval, many methods require the
    tensor to be invariant and invar == False is mainly meant for
    handling vectors of singular values and eigenvalues. If invar ==
    True then defval must be 0, unless the tensor is a scalar of charge
    0.

    Note that many of these rules are not constantly checked for and can
    be broken by the user. In such cases behavior of the class is not
    quaranteed. The method check_consistency can be used to check that
    the tensor conforms to this definition.

    Rem: to conform to this logic, a matrix is encoded as an MPS
    with trivial vertical space: heights = [h1,h2,...],[1],[h1,h2,...]

    """
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Constructors and copy constructors
    ##
    def __init__(self, shape, heights=None, hmax=None, sects=None,
                 dtype=np.float_, defval=0, charge=1, invar=True):
        """ Constructor: Although heights is a keyword argument to conform
        to the interface of the Tensor class, it must in fact be set.
        sects defaults to {}.
        """
        assert(heights is not None)
        shape = list(map(list, shape))
        heights = list(map(list, heights))
        if hmax is not None:
            assert(charge <= hmax)
            for hs in heights:
                for h in hs:
                    assert(h <= hmax)
        assert(type(self).check_heights_shape_match(heights, shape))
        if invar and charge != 1:
            assert(defval == 0)
        if sects is None:
            sects = {}

        self.defval = defval
        self.invar = invar
        self.charge = charge
        self.shape = list(map(list, shape))
        self.dtype = dtype
        self.heights = heights
        self.hmax = hmax
        self.sects = sects
        
    # the copy constructor of the class.
    ##
    copy = deepcopy
    __copy__ = copy
    
    ##
    def view(self):
        """ A view is otherwise independent but identical to the
        original, but its sects points to the same numpy arrays as the
        sects of the original. In other words changing a whole block is ok,
        but modifying a block in place modifies the original as well.
        """
        view = self.empty_like()
        # Note that this is a shallow copy.
        view.sects = self.sects.copy()
        return view    

    ##
    def diag(self):
        """Diag either maps a square matrix-like to a vector of its diagonals
        or a vector to diagonal square matrix-like.

        If the input is a vector (which may be non-invariant) with
        heights=[h], shape=[dim], then the output is an
        invariant matrix with heights=[h,[1],h], shape=[dim,[1],dim].

        If the input is a matrix it should be invariant and square in
        the sense that its two indices are compatible, i.e. could be
        contracted with each other.

        """
        assert(len(self.shape) == 1 or self.is_mat_like())
        if len(self.shape) == 1: # vector to matrix
            dim = self.shape[0]
            hs = self.heights[0]
            shape = [dim, [1], dim]
            heights = [hs, [1], hs]
            sects = {}
            for k,v in self.sects.items():
                new_k = (k[0], 1, k[0]) # just one element in keys, at 0.
                sects[new_k] = np.expand_dims(np.diag(v), axis=1)
            res = type(self)(shape, heights=heights, hmax=self.hmax,
                             sects=sects, dtype=self.dtype)
            return res
        else: # matrix to vector
            assert(self.invar)
            assert(self.compatible_indices(self, 0, 2))
            dim = self.shape[0]
            hs = self.heights[0]
            shape = [dim]
            heights = [hs]
            sects = {}
            for h in hs:
                try:
                    diag_block = self[(h, 1, h)]
                    sects[(h,)] = np.diag(diag_block)
                except KeyError:
                    # The diagonal block was not found, so we move on.
                    pass
            res = type(self)(shape, heights=heights, hmax=self.hmax,
                             sects=sects, dtype=self.dtype, invar=False)
            return res

    
    # Recall: meaning of classmethod.  python does not have overloading
    # like c++ so this used to define new constructors with different
    # arguments. cls holds class itself.

    ##
    @classmethod
    def eye(cls, dim, hs=None, hmax=None, dtype=np.float_):
        """ Outputs an identity tensor of shape=[dim,[1],dim],
        heights=[hs,[1],hs].
        """
        assert(cls.check_h_dim_match(hs, dim))
        dim = list(dim)
        hs = list(hs)
        sects = {}
        for i, h in enumerate(hs):
            sects[h, 1, h] = np.eye(dim[i], dtype=dtype).reshape(dim[i],1,dim[i])
        shape = [dim, [1], dim]
        heights = [hs, [1], hs]
        res = cls(shape, heights=heights, hmax=hmax, sects=sects,
                  dtype=dtype)
        return res

    ##
    @classmethod
    def initialize_with(cls, numpy_func, shape, *args,
                        heights=None, hmax=None, invar=True,
                        charge=1, **kwargs):
        """ initialize_with will be called with different numpy_funcs to
        create initializer functions such as zeros and random. It sets
        all the valid blocks of the new tensor to
        numpy_func(block_shape, *args, **kwargs).
        """
        shape = list(map(list, shape))
        heights = list(map(list, heights))
        assert(cls.check_heights_shape_match(heights, shape))

        # We use a fancy way of passing optional arguments to __init__
        # to avoid setting defaults in two places.
        opt_args = {"heights": heights, "hmax": hmax, "charge": charge,
                    "invar": invar}
        try:
            opt_args["dtype"] = kwargs["dtype"]
        except KeyError:
            pass
        res = cls(shape, **opt_args)

        dimcombs = itertools.product(*tuple(shape))
        hscombs = itertools.product(*tuple(heights))
        if shape:
            for hcomb, dcomb in zip(hscombs, dimcombs):
                if res.is_valid_key(hcomb):
                    res[tuple(hcomb)] = numpy_func(dcomb, *args, **kwargs)
        else:
            if res.charge == 0:
                res.defval = numpy_func((), *args, **kwargs)
        return res

    ##
    def empty_like(self):
        """ Initializes a tensor that is like a copy of self, but with
        an empty sects = {}.
        """
        res = type(self)(self.shape.copy(), heights=self.heights.copy(),
                         hmax=self.hmax, dtype=self.dtype,
                         defval=self.defval, invar=self.invar,
                         charge=self.charge)
        return res

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Methods for slicing, setting and getting elements
    
    # Sets all the values in sects to 'value'
    ##
    def fill(self, value):
        self.defval = value
        for v in self.sects.values():
            v.fill(value)

    # Defines what self[k] means
    ##
    def __getitem__(self, k):
        """ If self[k] is called, then first self.sects[k] is checked.
        If the key is not found, we check if the k still is a valid key
        for this tensor. If yes, a block full of defval is created, set
        to be self[k] and returned. If not, a KeyError is raised, with
        message describing what went wrong.
        """
        try:
            return self.sects.__getitem__(k)
        except KeyError:
            if not isinstance(k, tuple) or not len(k) == len(self.heights):
                raise KeyError("Malformed block key: %s"%str(k))
            if self.is_valid_key(k):
                # Even though the requested block was not found it's a
                # valid block, so we create it.
                try:
                    block = self.defblock(k)
                except ValueError:
                    raise KeyError("Requested block has non-existent height.")
                self[k] = block
                return block
            else:
                raise KeyError("Requested a block forbidden by fusion: %s"
                               % str(k))

    ##
    def value(self):
        """ For a scalar tensor, return the scalar. """
        if self.shape:
            raise ValueError("value called on a non-scalar tensor.")
        else:
            return self.defval

    ##
    def __setitem__(self, key, value):
        return self.sects.__setitem__(key, value)
        
    ##
    def __delitem__(self, key):
        return self.sects.__delitem__(key)
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Operator methods

    # Use print(str(T)) to see what's inside the object T.  repr
    # and str are the official and informal string representation of
    # objects in python
    
    ##
    def __repr__(self):
        r = ("%r(%r, heights=%r, hmax=%r, sects=%r, dtype=% rdefval=%r, "
             "invar=%r, charge=%r"%(
                 type(self), self.shape, self.heights, self.hmax, self.sects,
                 self.dtype, self.defval, self.invar, self.charge))
        return r

    ##
    def __str__(self, *args, **kwargs):
        r = ("%s object:\n"
             "shape = %s,\n"
             "heights = %s,\n" 
             "hmax = %s, charge = %s\n"
             "defval = %s, invar = %s, dtype = %s\n"
             "blocks:")%(str(type(self)), self.shape,
                           self.heights, self.hmax, self.charge, 
                           self.defval, self.invar, self.dtype)
        for k,v in self.sects.items():
            r += "\n%s:\n%s"%(k,v)
        return r

    # The operator module exports a set of efficient functions
    # corresponding to the intrinsic operators of Python.  Names with
    # leading and trailing underscores are reserved for python itself.

    ##
    __add__ = generate_binary_deferer(operator.add)
    __sub__ = generate_binary_deferer(operator.sub)
    __mul__ = generate_binary_deferer(operator.mul)
    __truediv__ = generate_binary_deferer(operator.truediv)
    __floordiv__ = generate_binary_deferer(operator.floordiv)
    __mod__ = generate_binary_deferer(operator.mod)
    __divmod__ = generate_binary_deferer(divmod)
    __pow__ = generate_binary_deferer(pow)
    __lshift__ = generate_binary_deferer(operator.lshift)
    __rshift__ = generate_binary_deferer(operator.rshift)
    __and__ = generate_binary_deferer(operator.and_)
    __xor__ = generate_binary_deferer(operator.xor)
    __or__ = generate_binary_deferer(operator.or_)

    def arg_swapper(op):
        def res(a,b, *args, **kwargs):
            return op(b,a, *args, **kwargs)
        return res

    __radd__ = generate_binary_deferer(arg_swapper(operator.add))
    __rsub__ = generate_binary_deferer(arg_swapper(operator.sub))
    __rmul__ = generate_binary_deferer(arg_swapper(operator.mul))
    __rtruediv__ = generate_binary_deferer(arg_swapper(operator.truediv))
    __rfloordiv__ = generate_binary_deferer(arg_swapper(operator.floordiv))
    __rmod__ = generate_binary_deferer(arg_swapper(operator.mod))
    __rdivmod__ = generate_binary_deferer(arg_swapper(divmod))
    __rpow__ = generate_binary_deferer(arg_swapper(pow))
    __rlshift__ = generate_binary_deferer(arg_swapper(operator.lshift))
    __rrshift__ = generate_binary_deferer(arg_swapper(operator.rshift))
    __rand__ = generate_binary_deferer(arg_swapper(operator.and_))
    __rxor__ = generate_binary_deferer(arg_swapper(operator.xor))
    __ror__ = generate_binary_deferer(arg_swapper(operator.or_))

    __eq__ = generate_binary_deferer(operator.eq)
    __ne__ = generate_binary_deferer(operator.ne)
    __lt__ = generate_binary_deferer(operator.lt)
    __le__ = generate_binary_deferer(operator.le)
    __gt__ = generate_binary_deferer(operator.gt)
    __ge__ = generate_binary_deferer(operator.ge)

    __neg__ = generate_unary_deferer(operator.neg)
    __pos__ = generate_unary_deferer(operator.pos)
    __abs__ = generate_unary_deferer(abs)
    __invert__ = generate_unary_deferer(operator.invert)

    ##
    def conj(self):
        """ Conjugation complex conjugates only since self-dual charges.
        """
        res = self.defer_unary_elementwise(np.conj)
        return res

    ##
    def astype(self, dtype, casting='unsafe', copy=True):
        if not np.can_cast(self.dtype, dtype, casting=casting):
            raise ValueError("Cannot cast {} into {} with casting={}.".
                             format(self.dtype, dtype, casting))
        if copy:
            res = self.copy()
        else:
            res = self
        res.dtype = dtype
        for k, v in res.sects.items():
            res[k] = v.astype(dtype, casting=casting, subok=True, copy=False)
        return res

    ##
    conjugate = conj
    sqrt = generate_unary_deferer(np.sqrt)
    sign = generate_unary_deferer(np.sign)
    log = generate_unary_deferer(np.log)
    exp = generate_unary_deferer(np.exp)
    abs = __abs__

    # Act on each element - sector - of the tensor with the
    # operator op_func. First create an empty like object and then act
    # and insert in sects. unary means on the single self object.

    ##
    def defer_unary_elementwise(self, op_func, *args, **kwargs):
        """ Produces a new tensor that is like self, but all the blocks
        v have been acted on with op_func(v, *args, **kwargs), as has
        the defval. If defval ends up being mapped to something non-zero
        then the resulting tensor is not invariant and is flagged as
        such.

        This method can be used to create basic element-wise unary
        operations on tensors, such as negation and element-wise
        absolute value.
        """
        res = self.empty_like()
        res.defval = op_func(self.defval, *args, **kwargs)
        if res.defval != 0:
            res.invar = False
        for k,v in self.sects.items():
            res_block = op_func(v, *args, **kwargs)
            res.sects[k] = res_block
        return res

    # Operations done on self and B, two tensors. Used eg in add
    # and comparisons. One can sum also non invariant tensors.
    ##
    def defer_binary_elementwise(self, B, op_func, *args, **kwargs):
        """ If both self and B are SU2kMPS, then their blocks and
        defvals are operated on pair-wise with op_func(_, _, *args,
        **kwargs). The two tensors should in this case be of the same
        form: same qnums, dims, etc. If not, either warnings or errors
        are raised, depending on whether the operation can still be
        carried out meaningfully.
        
        If B is not an SU2kMPS then all the blocks and
        the defval of self are operated on with op_func(_, B, *args,
        **kwargs).

        The operation is never in-place, but always a new tensor. The
        new tensor is like self in its attributes, but may be
        non-invariant if its defval ends up being non-zero.

        This method can be used to create element-wise binary operations
        on tensors, such as basic arithmetic and comparisons.
        """
        try:
            res_dtype = np.result_type(self.dtype, B.dtype)
        except AttributeError:
            res_dtype = np.result_type(self.dtype, B)
        res = self.empty_like()
        res.dtype = res_dtype
        if isinstance(B, SU2kMPS):
            # self and B should be of the same form or one should be a
            # scalar. They should also have the same hmax.
            assert(type(self).check_form_match(tensor1=self, tensor2=B)
                    or not self.shape or not B.shape)
            assert(self.hmax == B.hmax)
            # They may have different charges but this
            # generates a warning.
            if self.charge != B.charge and self.shape:
                warnings.warn("Binary operation called on non-scalar tensors "
                              "with differing charges (%i and %i)."
                              %(self.charge, B.charge), stacklevel=3)

            # Checks are done, move on to operating.
            res.defval = op_func(self.defval, B.defval, *args, **kwargs)
            all_keys = set().union(self.sects, B.sects)
            for k in all_keys:
                # Use B[k] and self[k], but default to defval if key not
                # found.
                a = self.sects.get(k, self.defval)
                b = B.sects.get(k, B.defval)
                res_block = op_func(a, b, *args, **kwargs)
                res.sects[k] = res_block
        else:
            res.defval = op_func(self.defval, B, *args, **kwargs)
            for k,v in self.sects.items():
                res_block = op_func(v, B, *args, **kwargs)
                res.sects[k] = res_block
        if (res.shape or res.charge) and res.defval != 0:
            res.invar = False
        return res

    ##
    def any(self):
        """ Check whether any of the elements of the tensor are True,
        i.e. not equal to zero.
        """
        for v in self.sects.values():
            if np.any(v):
                return True
        if self.is_full():
            return False
        else:
            return np.any(self.defval)
            
    ##
    def all(self):
        """ Check whether all of the elements of the tensor are True.
        """
        for v in self.sects.values():
            if not np.all(v):
                return False
        if self.is_full():
            return True
        else:
            return np.all(self.defval)

    ##
    def allclose(self, B, rtol=1e-05, atol=1e-08):
        """ Check whether all of the elements of the tensors are close
        to each other. See numpy.allclose for explanations of the
        tolerance arguments.
        """
        # self and B should be of the same form and have the same
        # hmax.
        assert(type(self).check_form_match(tensor1=self, tensor2=B))
        assert(self.hmax == B.hmax)

        # Form checks done, move on to comparing blocks.
        # if repeated sects, take just one.
        all_keys = set().union(self.sects, B.sects)
        for k in all_keys:
            # 2nd argument of get is default value, if key does not exit
            a = self.sects.get(k, self.defval) 
            b = B.sects.get(k, B.defval)
            if not np.allclose(a,b):
                return False
        if self.is_full():
            return True
        else:
            return np.allclose(self.defval, B.defval)

    ##
    def max(self):
        if 0 in type(self).flatten_shape(self.shape):
            raise ValueError("zero-size array has no maximum")
        if not self.shape:
            return self.defval
        # If not all blocks are set, then the tensor has an element of
        # defval somewhere.
        m = -np.inf if self.is_full() else self.defval
        for v in self.sects.values():
            try:
                n = np.max(v)
            except ValueError:
                # This block was zero-size.
                n = m
            m = max(m, n)
        return m

    ##
    def min(self):
        if 0 in type(self).flatten_shape(self.shape):
            raise ValueError("zero-size array has no maximum")
        if not self.shape:
            return self.defval
        # If not all blocks are set, then the tensor has an element of
        # defval somewhere.
        m = np.inf if self.is_full() else self.defval
        for v in self.sects.values():
            try:
                n = np.min(v)
            except ValueError:
                # This block was zero-size.
                n = m
            m = min(m, n)
        return m

    ##
    def average(self):
        s = self.sum()
        flat_shape = self.flatten_shape(self.shape)
        num_of_elements = reduce(operator.mul, flat_shape, 1)
        average = s/num_of_elements
        return average

    ##
    def real(self):
        res = self.defer_unary_elementwise(np.real)
        res.dtype = np.float_
        return res

    ##
    def imag(self):
        res = self.defer_unary_elementwise(np.imag)
        res.dtype = np.float_
        return res

    ##
    def sum(self):
        if self.shape:
            if self.defval:
                raise NotImplementedError("Sum of defval != 0 not "
                                          "implemented.")
            s = 0
            for v in self.sects.values():
                s += np.sum(v)
        else:
            s = self.defval
        return s

    ##
    def __len__(self):
        """ This works as for numpy arrays: len returns the total
        dimension of the first leg.
        """
        return self.flatten_dim(self.shape[0])

    ##
    def __bool__(self):
        if self.shape:
            raise ValueError("The truth value of a tensor with more than one "
                             "element is ambiguous. Use a.any() or a.all()")
        else:
            return bool(self.defval)


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # To and from normal numpy arrays

    ##
    def to_ndarray(self):
        """ Returns a corresponding numpy array. The order of the blocks
        in the result is such that along every index the blocks are
        organized according to rising heights.
        """
        ndshape = type(self).flatten_shape(self.shape)
        res = np.full(ndshape, self.defval, dtype=self.dtype)
        if 0 in ndshape:
            return res
        shp, qhp = type(self).sorted_shape_heights(tensor=self)
        # ranges is like shape, but every number d is replaced by a
        # tuple (a,a+d) where a is the sum of all the previous entries
        # in the same dim.
        ranges = []
        for dim in shp:
            prv = dim[0]
            r = [(0, prv)]
            for d in dim[1:]:
                nxt = prv + d
                r.append((prv, nxt))
                prv = nxt
            ranges.append(r)
        for k, v in self.sects.items():
            slc = ()
            for i, qnum in enumerate(k):
                r = ranges[i][qhp[i].index(qnum)]
                slc += (slice(r[0], r[1]),)
            res[slc] = v
        return res

    # This is a method to construct an invariant abelian tensor out of
    # a normal array
    ##
    @classmethod
    def from_ndarray(cls, a, shape=None, heights=None, hmax=None,
                     invar=True, charge=1):
        """ Takes an ndarray a, and maps it to a corresponding
        SU2kMPS of the form given in the other arguments. Although
        shape and heights are keyword arguments to maintain a common
        interface with Tensor, they are not optional. The blocks are
        read in the same order as they are written in to_ndarray, i.e.
        rising heights along every leg. Note hence that the ordering of the
        heights given has no effect.
        """
        is_bool = a.dtype == np.bool_
        invar = invar and not is_bool
        shape, heights = cls.sorted_shape_heights(shape=shape, heights=heights)
        # this should create an empty tensor
        res = cls(shape, heights=heights, hmax=hmax, dtype=a.dtype,
                  invar=invar, charge=charge)
        if not a.shape:
            res.defval = a
            return res
        # ranges is like shape, but every number d is replaced by a
        # tuple (a,a+d) where a is the sum of all the previous entries
        # in the same dim.
        ranges = []
        for dim in res.shape:
            prv = dim[0]
            r = [(0, prv)]
            for d in dim[1:]:
                nxt = prv + d
                r.append((prv, nxt))
                prv = nxt
            ranges.append(r)
        for k in itertools.product(*res.heights):
            # insert only valid keys
            if res.is_valid_key(k):
                slc = ()
                for i, qnum in enumerate(k):
                    r = ranges[i][res.heights[i].index(qnum)]
                    slc += (slice(r[0], r[1]),)
                block = a[slc]
                res.sects[k] = block
        return res

    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Miscellaneous

    ##
    def form_str(self):
        s = "shape: %s\nheights: %s"%(str(self.shape), str(self.heights))
        return s

    ##
    @staticmethod
    def flatten_shape(shape):
        try:
            return tuple(map(SU2kMPS.flatten_dim, shape))
        except TypeError:
            return shape

    ##
    @staticmethod
    def flatten_dim(dim):
        try:
            return sum(dim)
        except TypeError:
            return dim

    ##
    def norm_sq(self):
        conj = self.conj()
        all_inds = tuple(range(len(self.shape)))
        norm_sq = self.dot(conj, (all_inds, all_inds))
        return np.abs(norm_sq.value())

    ##
    def norm(self):
        return np.sqrt(self.norm_sq())

    ##
    def transpose(self, p=(1,0)):
        """ Transposes legs of self, returns a view. """
        res = self.empty_like()
        for k,v in self.sects.items():
            kt = tuple(map(k.__getitem__, p))
            res.sects[kt] = v.transpose(p)
        res.shape = list(map(self.shape.__getitem__, p))
        res.heights = list(map(self.heights.__getitem__, p))
        return res

    ##
    def is_valid_key(self, key):
        """ Returns true if self.invar is not True or key is a key for a
        valid block allowed by fusion. Otherwise False.
        """
        if not self.invar:
            return True
        if len(key) != len(self.heights):
            return False
        hmax = self.hmax
        # go through vert lines (odd heights) and check (N_{a_i})_{h_{i-1},h_i}=1
        list_as = key[1::2]
        x = True
        for i,a in enumerate(list_as):
            x = x and SU2k_data.Nmat_el(self.hmax,key[2*i],key[2*i+1],key[2*i+2])==1
        return x

    ##
    def compatible_indices(self, other, i, j):
        """ Returns True if leg i of self may be contracted with leg j
        of other, False otherwise. 

        """
        s_dim = self.shape[i] 
        s_qim = self.heights[i]
        o_dim = other.shape[j] 
        o_qim = other.heights[j]
        o_qimdim = set(zip(o_qim, o_dim))
        s_qimdim = set(zip(s_qim, s_dim))
        res = o_qimdim == s_qimdim
        return res

    ##
    def is_mat_like(self):
        """ Returns True if self is mat-like
        """
        x = self.shape[1] == [1]
        y = self.heights[1] == [1]
        return x and y

    ##
    def is_2_sites(self):
        """ Returns True if self is 2 sites MPS

        """
        x = len(self.shape)==5
        x = x and self.charge == 1
        x = x and self.invar
        x = x and self.shape[1]==[1] and self.shape[3]==[1]
        x = x and self.heights[1]==[2] and self.heights[3]==[2]
        x = x and self.shape[2] == [1]*len(self.shape[2])
        x = x and self.defval==0
        return x 

    ##
    def is_wf(self):
        """ Returns True if self is a valid SU2kMPS wavefunction

        """
        # some trivial checks
        x = self.charge == 1
        x = x and self.invar
        x = x and self.defval==0
        # length is an arbitrary odd int
        L = len(self.shape)
        x = x and L % 2 == 1
        # check that odd heights are the same and equal to 2, and
        # their shape is [1]
        for l in range(1,L,2):
            sh = self.shape[l]
            hs = self.heights[l]
            x = x and sh == [1]
            x = x and hs == [2] 
        # check that intermidiate even heights have shape [1,...,1]
        for l in range(2,L-1,2):
            sh = self.shape[l]
            x = x and sh == [1]*len(sh)
        return x 
    
    ##
    def expand_dims(self, axis):
        """ Returns a view of self that has an additional leg at the
        position axis. This leg has only one height, 1, and dimension 1.
        """
        res = self.empty_like()
        res.shape.insert(axis, [1])
        res.heights.insert(axis, [1])
        if self.shape:
            for k,v in self.sects.items():
                new_k = list(k)
                new_k.insert(axis, 0)
                res[tuple(new_k)] = np.expand_dims(v, axis)
        elif res.charge == 1:
            res[(1,)] = np.array((res.defval,), dtype=res.dtype)
            res.defval = 0
        return res
    
    ##
    @staticmethod
    def sorted_shape_heights(tensor=None, shape=None, heights=None):
        """ Sort shape and heights according to ascending heights along every
        leg. Used by to_ and from_ndarray.
        """
        shape = tensor.shape if shape is None else shape
        heights = tensor.heights if heights is None else heights
        sorted_qhp = []
        sorted_shp = []
        for qim, dim in zip(heights, shape):
            qim, dim = zip(*sorted(zip(qim,dim)))
            sorted_qhp.append(qim)
            sorted_shp.append(dim)
        return sorted_shp, sorted_qhp

    ##
    def defblock(self, key):
        """ Returns an ndarray of the size of the block self[key],
        filled with self.defval.  This works regardless of whether
        self[key] is set or not and whether the block is allowed by
        symmetry.
        """
        block_shape = []
        for i,qnum in enumerate(key):
            block_shape.append(self.shape[i][self.heights[i].index(qnum)])
        block = np.full(block_shape, self.defval, dtype=self.dtype)
        return block

    ##
    def is_full(self):
        """ Returns True if the elements in self.sects cover all the
        elements in self.
        """
        elements_in_sects = sum(map(operator.attrgetter("size"),
                                    self.sects.values()))
        elements_in_total = reduce(operator.mul,
                                   type(self).flatten_shape(self.shape),
                                   1)
        res = elements_in_sects >= elements_in_total
        return res

    ##
    def check_consistency(self):
        """ Checks that self conforms to the defition given in the
        documentation of the class. If yes, returns True, otherwise
        raises an assertion error.  This method is meant to be used by
        the user (probably for debugging) and is not called anywhere in
        the class.
        """
        try:
            assert(len(self.shape) == len(self.heights))
            assert(all((len(dim) == len(qim))
                   for dim,qim in zip(self.shape, self.heights)))
            # Check that every sect has a valid key and the correct
            # shape and dtype.
            for k,v in self.sects.items():
                assert(v.dtype == self.dtype)
                assert(self.is_valid_key(k))
                block_shp_real = v.shape
                qnum_inds = tuple(self.heights[i].index(qnum)
                                  for i, qnum in enumerate(k))
                block_shp_claimed = tuple([self.shape[i][j]
                                          for i, j in enumerate(qnum_inds)])
                assert(block_shp_claimed == block_shp_real)
            # Other checks.
            if self.invar and (self.charge != 1 or self.shape):
                assert(self.defval == 0)
        except:
            raise
        return True

    ##
    @classmethod
    def check_h_dim_match(cls, hs, dim):
        """ Check that the given heights hs and dim match, i.e. are valid for
        the same leg.
        """
        return len(hs) == len(dim)

    ##
    @classmethod
    def check_heights_shape_match(cls, heights, shape):
        """ Check that the given heights and shape match, i.e. are valid
        for the same tensor.
        """
        return all(cls.check_h_dim_match(hs, dim)
                   for hs,dim in zip(heights,shape))
    
    ##
    @classmethod
    def check_form_match(cls, tensor1=None, tensor2=None,
                         heights1=None, shape1=None, 
                         heights2=None, shape2=None,
                         hmax=None):
        """ Check that the given two tensors have the same form in the
        sense that both tensors have the same heights for the same
        indices and with the same dimensions. In stead of giving two
        tensors, sets of heights, shapes and a hmax can also
        be given.
        """

        if tensor1 is not None:
            heights1 = tensor1.heights
            shape1 = tensor1.shape
        if tensor2 is not None:
            heights2 = tensor2.heights
            shape2 = tensor2.shape
 
        if not (len(heights1) == len(heights2) == len(shape1) == len(shape2)):
            return False
        for qim1, dim1, qim2, dim2 in zip(heights1, shape1,
                                          heights2, shape2):
            # This is almost like compatible_indices, but for the
            # missing minus sign when building o_qim.
            qimdim1 = set(zip(qim1, dim1))
            qimdim2 = set(zip(qim2, dim2))
            if not qimdim1 == qimdim2:
                return False
        return True

    ##
    @classmethod
    def find_trunc_dim(cls, S, S_sects, minusabs_next_els, dims,
                       chis=None, eps=0, break_degenerate=False,
                       degeneracy_eps=1e-6, norm_type="frobenius",
                       remove_small=False):
        """S is a vector containing all the singular values, S_sects is a
        dictionary whose keys are the sectors and values are the
        triples (s,u,v).  minusabs_next_els is a heapq whose nodes are
        (-largest sing_val for sect k, k) ordered such that - the largest is
        at the root. dims = dict with values the dimensions for each key.

        Returns chi (the truncation dimension, which is reduced to
        avoid breaking degeneracies if break_degenerate=False)
        dims (see above), rel_err=sqrt(sum(S^2[chi:])/sum(S^2))

        """
        # First, find what chi will be.
        S = -np.sort(-np.abs(S)) # sort so that -biggest first.
#        print("In find_trunc_dim: S =",S)
        if norm_type=="frobenius":
            # later take square root
            S = S**2
            eps = eps**2
        else:
            raise ValueError("Unknown norm_type {}".format(norm_type))
        sum_all = np.sum(S) # = sum(S**2)
        if sum_all != 0:
            for chi in chis:
                if not break_degenerate:
                    # Make sure that we don't break degenerate singular
                    # values by including one but not the other.
                    while 0 < chi < len(S):
                        last_eig_in = S[chi-1]
                        last_eig_out = S[chi]
                        rel_diff = np.abs(last_eig_in-last_eig_out)/last_eig_in
                        if rel_diff < degeneracy_eps:
                            chi -= 1
                        else:
                            break
                sum_disc = sum(S[chi:])
                rel_err = sum_disc/sum_all # relative error
                if rel_err <= eps: # recall, also eps is squared so ok comparison
                    break
            if norm_type=="frobenius":
                rel_err = np.sqrt(rel_err)
        else:
            rel_err = 0
            chi = min(chis)
        # discard very small singular values since then we need to
        # invert Lambda and these will not affect the result (due to
        # instability)
        if remove_small:
            tol = 10.**(-12)
            nz_S = np.sum(np.sqrt(S)>tol)
            chis.append(nz_S)
            chi = min(chis)
        # Find out which eigenvalues to keep.
        dim_sum = 0
        while(dim_sum < chi):
            # index_to_add is the index of the block which has
            # the largest eigenvalue that is not yet
            # included in truncation.
            try:
                # get from the heap - biggest sing value and its key
                minusabs_el_to_add, key = heapq.heappop(minusabs_next_els)
            except IndexError:
                # All the dimensions are fully included.
                break
            dims[key] += 1
            this_key_els = S_sects[key][0] # sing value for this key
            if dims[key] < len(this_key_els):
                # take next element and push it to the heap
                next_el = this_key_els[dims[key]]
                heapq.heappush(minusabs_next_els, (-np.abs(next_el), key))
            dim_sum += 1
        return chi, dims, rel_err
    
    ##
    def matrix_decomp_format_chis(self, chis, eps):
        """ A function for formatting the truncation parameters
        of SVD and eig. This is meant to be called by the matrix_svd and
        matrix_eig functions of subclasses.
        """
        if chis is None:
            # note shape[0] and [2] are the non-trivial dimensions of
            # a matrix-like MPS
            min_dim = min(type(self).flatten_dim(self.shape[0]),
                          type(self).flatten_dim(self.shape[2])) + 1
            if eps > 0:
                chis = tuple(range(min_dim))
            else:
                chis = [min_dim]
        else:
            try:
                chis = tuple(chis)
            except TypeError:
                chis = [chis]
            if eps == 0:
                chis = [max(chis)]
            else:
                chis = sorted(chis)
        return chis
    
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # The meat: actual tensor operations

    def fuse(self, inds=None, erase_inds=False):
        """Joins indices together in the spirit of reshape. inds=(i,j),
        and only the following possibility are considered:
        i=0,j=1 or j=L-2,i=L-1 where L is the number of indices,
        and the shape of j can only be = [1]. 
        
        The method returns the fused MPS where the j index has height
        = 1 if remove_inds=False. If True, removes i and j.
        
        The method does not modify the original tensor, nor its length.

        """
        if inds==None:
            return self.view()
        # Check format:
        i = inds[0]
        j = inds[1]
        L = len(self.shape)
        if L < 3:
            print("In SU2kMPS.fuse: only deal with L>=3.")
            return self.view()
        if (i != 0 and i != L-1) or (j != 1 and j != L-2) or self.shape[j] != [1]:
            print("In SU2kMPS.fuse: case not covered inds = ", inds)
            return self.view()
        # next_pos = index where next horiz height is
        if i == 0: # means i=0,j=1
            next_pos = 2 
        else: # i =L-1
            next_pos = L-3
        # Go through every valid index instead of every key in sects,
        # because blocks of zeros may be concatenated with other blocks.
        valid_ks = (hcomb for hcomb in itertools.product(*self.heights)
                    if self.is_valid_key(hcomb))
        # here we initialize the new sectors. the only new admissible
        # configs are [h,1,h,...] if i=0 or [...,h,1,h] if i=L-1.
        new_sects = {}
        # order keeps memory of values hi contributing to new_k
        order = {}
        for k in valid_ks:
            v = self[k]
            hi = k[i] 
            new_k = list(k)
            if erase_inds:
                if i == 0:
                    new_k = new_k[2:]
                else: # i = L-1
                    new_k = new_k[:L-2]
                # avoid reshaping v at this point. It will be done after
            else:
                new_k[i] = k[next_pos] 
                new_k[j] = 1
            new_k = tuple(new_k)  
            if new_k in new_sects:
                # already exists, concatenate at the end, i is the axis
                new_sects[new_k] = np.concatenate((new_sects[new_k], v), axis = i)
                order[new_k].append(hi)
            else:
                new_sects[new_k] = v
                order[new_k] = [hi]
        
        # transpose according to canonical order:
        for new_k in new_sects:
            cur_order = order[new_k]
            # Choose as canonical order the fusion range. This contains
            # all the hi's which fusing with height[j][0] give new_hi.
            # remember that we already erased k.
            if erase_inds and i==L-1:
                new_hi = new_k[-1]
            else:
                new_hi = new_k[i]
            can_order = SU2k_data.fusion_range(self.hmax,self.heights[j][0],new_hi)
            can_order =  [x for x in can_order if x in self.heights[i]]
            if cur_order == list(can_order):
#                print("In SU2kMPS.fuse: order match")
                # if erase indices, just reshape
                if erase_inds:
                    v = new_sects[new_k]
                    sh = list(v.shape)
                    sh[next_pos] *= sh[i]
                    if i == 0:
                        sh = sh[next_pos:]
                    else: # i = L-1
                        sh = sh[:j]
                    new_sects[new_k] = v.reshape(sh)
                else: #nothing, just continue
                    continue
            else: # rearrange and if erase indices, change shape, TODO: check
                print("**In SU2kMPS.fuse: key, order not match, rearrange:")
#                print(cur_order, list(can_order))
                v = new_sects[new_k]
                sh = v.shape
                # shape of the chunck that we have to rearrange
                old_sh_i = self.shape[i]
                old_h_i = self.heights[i]
                chunk_sh = [old_sh_i[old_h_i.index(x)] for x in cur_order]
                # determine permutation of entries of 0-th index of v
                perm = []
                for l in can_order:
                    # start at the sum all shapes before it
                    pos_l = cur_order.index(l)
                    start = sum(chunk_sh[:pos_l])
                    end = start + chunk_sh[pos_l]
                    perm += list(range(start,end))
                if i == 0:
                    v_new = v[perm,:]
                    # if erase indices, change shape
                    if erase_inds:
                        sh = list(sh)
                        sh[next_pos] *= sh[i]
                        sh = sh[next_pos:]
                        sh = tuple(sh)
                        v_new = v_new.reshape(sh)
                else: # i = L-1 
                    # use rollaxis to bring the axis i to 0-th pos,
                    # permute and then roll back
                    v_new = np.rollaxis(v,i)
                    v_new = v_new[perm,:]
                    v_new = np.rollaxis(v_new, 0, i+1)
                    # if erase indices, change shape
                    if erase_inds:
                        sh = list(sh)
                        sh[next_pos] *= sh[i]
                        sh = sh[:j]
                        sh = tuple(sh)
                        v_new = v_new.reshape(sh)
                # finally assign:
                new_sects[new_k] = v_new

        # Compute the new shape and heights
        nsd = {} # new shape dictionary
        for n, h in zip(self.shape[i], self.heights[i]):
            # self.heights[j][0] has size 1
            for new_h in SU2k_data.fusion_range(self.hmax, h, self.heights[j][0]):
                if new_h in nsd:
                    nsd[new_h] += n
                else:
                    nsd[new_h] = n
        # assign
        tmp_heights = list(nsd.keys())
        tmp_shape = list(nsd.values())
        tmp_shape, tmp_heights = self.sorted_shape_heights(shape=[tmp_shape], 
                                                           heights=[tmp_heights])
        # TODO: tmp_heights = self.heights[next_pos]
        res = self.empty_like()
        if erase_inds:
            # element wise product of elements in lists:
            res.shape[next_pos] = [a*b for a,b in zip(list(tmp_shape[0]),res.shape[next_pos])]
            if i == 0:
                res.shape = res.shape[next_pos:]
                # just delete spurious heights:
                res.heights = res.heights[next_pos:]
            else: # i = L-1
                res.shape = res.shape[:j]
                # just delete spurious heights:
                res.heights = res.heights[:j]
            if len(res.shape)==1:
                # if left with just one index, invar=False
                res.invar=False
        else:
            res.shape[i] = list(tmp_shape[0])
            res.shape[j] = [1]
            res.heights[i] = list(tmp_heights[0])
            res.heights[j] = [1]
        res.sects = new_sects

        return res

    #
    def split(self, inds=None, new_dim_i=[], new_dim_j=[], new_h_i=[],
              new_h_j=[]):
        """Split inds in the spirit of reshape.  Here inds is a couple
        of indices, inds=(i,j) which are either i=0,j=1 or
        i=L-1,j=L-2.  This routine applies only to case
        heights[j]=[1], in which case the returned MPS has: shape[i,j]
        = new_dim_i, new_dim_j ; heights[i,j] = new_h_i, new_h_j
        
        Schematically:
        new_sects[order[k],new_h_j,...]=sects[k][new_dim_i[order[k]]
        order is the canonical order given by fusion_range

        The method returns the split MPS and is understood to be used
        on MPS which have been processed previously by fuse.
        
        The method does not modify the original tensor.

        """
        # Check format:
        i = inds[0]
        j = inds[1]
        L = len(self.shape)
        if L < 3:
            print("In SU2kMPS.split: only deal with L>=3.")
            return self.view()
        if (i != 0 and i != L-1) or (j != 1 and j != L-2) or self.shape[j] != [1]:
            print("In SU2kMPS.split: case not covered inds = ", inds)
            return self.view()
        assert(len(new_h_j)==1 and new_dim_j==[1])

        # here we initialize the new sectors: 
        new_sects = {}
        for k,v in self.sects.items():
            a = new_h_j[0]# j is 1d
            can_order = SU2k_data.fusion_range(self.hmax,a,k[i])
            can_order = [x for x in can_order if x in new_h_i]
            # position the axis i at 0
            v=np.rollaxis(v,i)
            start = 0
            for l in can_order:
                new_k = list(k)
                new_k[i] = l
                new_k[j] = a
                new_k = tuple(new_k)
                end = start + new_dim_i[new_h_i.index(l)]
                # slice along axis 0 which is i since rolled
                tmp_sect = v[start:end]
                # rollaxis back
                new_sects[new_k] = np.rollaxis(tmp_sect, 0, i+1)
                start = end

        # Compute the new shape and heights
        res = self.empty_like()
        res.shape[i] = new_dim_i
        res.shape[j] = new_dim_j
        res.heights[i] = new_h_i
        res.heights[j] = new_h_j
        res.sects = new_sects

        return res

    ##
    def matrix_dot(self, other, transpose_self=False, transpose_other=False):
        """Takes the dot product of either two matrix-like SU2kMPS or an
        SU2kMPS and an SU2kMPS vector. If self or other is a matrix,
        it must be invariant and have defval == 0.
        If transpose=True, just multiply with the transpose

        vec . vect -> scalar
        vec . mat-like, vec . mat-like -> vec
        mat . mat -> mat

        """
        assert(self.hmax == other.hmax)

        # The following essentially a massive case statement on whether
        # self and other are scalar, vectors or matrices. Unwieldly, but
        # efficient and clear.
        if not self.shape and not other.shape:
            return self*other
        else:
            res_dtype = np.result_type(self.dtype, other.dtype)
            assert(self.charge == 1 and other.charge == 1) # we do not want to handle other cases.
            res_charge = 1
            res_invar = self.invar and other.invar

            # Vector times vector -> scalar
            if len(self.shape) == 1 and len(other.shape) == 1:
                assert(self.compatible_indices(other, 0, 0))
                res = 0
                for h in self.heights[0]:
                    try:
                        a = self[(h,)]
                        b = other[(h,)]
                    except KeyError:
                        continue
                    prod = np.dot(a,b)
                    if prod:
                        res += np.dot(a,b)
                res = type(self)([], heights=[], hmax=self.hmax, sects={},
                                 defval=res, dtype=res_dtype,
                                 charge=res_charge, invar=res_invar)
            else:
                res_sects = {}

                # Vector times matrix
                if len(self.shape) == 1:
                    assert(other.invar)
                    assert(other.defval == 0)
                    assert(other.is_mat_like())
                    if transpose_other:
                        ind_other = 2
                        rem_other = 0
                    else:
                        ind_other = 0
                        rem_other = 2
                    assert(self.compatible_indices(other, 0, ind_other))
                    res_shape = [other.shape[rem_other]]
                    res_heights = [other.heights[rem_other]]
                    for h in self.heights[0]:
                        try:
                            a = self[(h,)]
                            b = other[(h,1,h)]
                            tmp = np.tensordot(a,b, axes=(0,ind_other))
                            res_sects[(h,)] = tmp.reshape(b.shape[rem_other])
                        except KeyError:
                            # One of the blocks was zero so the resulting
                            # block will be zero.
                            continue

                # Matrix times vector
                elif len(other.shape) == 1:
                    assert(self.invar)
                    assert(self.defval == 0)
                    assert(self.is_mat_like())
                    if transpose_self:
                        ind_self = 0
                        rem_self = 2
                    else:
                        ind_self = 2
                        rem_self = 0
                    assert(self.compatible_indices(other, ind_self, 0))
                    res_shape = [self.shape[rem_self]]
                    res_heights = [self.heights[rem_self]]
                    for h in self.heights[2]:
                        try:
                            a = self[(h,1,h)]
                            b = other[(h,)]
                            tmp = np.tensordot(a,b, axes=(ind_self,0))
                            res_sects[(h,)] = tmp.reshape(a.shape[rem_self])
                        except KeyError:
                            # One of the blocks was zero so the resulting
                            # block will be zero.
                            continue

                # Matrix times matrix
                else:
                    assert(self.invar and other.invar)
                    assert(self.defval == other.defval == 0)
                    assert(self.is_mat_like() and other.is_mat_like())
                    if transpose_self and transpose_other:
                        ind_self = 0
                        rem_self = 2
                        ind_other = 2
                        rem_other = 0
                    elif transpose_self and not transpose_other:
                        ind_self = 0
                        rem_self = 2
                        ind_other = 0
                        rem_other = 2
                    elif transpose_other and not transpose_self:
                        ind_self = 2
                        rem_self = 0
                        ind_other = 2
                        rem_other = 0
                    else: #both not transpose
                        ind_self = 2
                        rem_self = 0
                        ind_other = 0
                        rem_other = 2
                    assert(self.compatible_indices(other, ind_self, ind_other))
                    res_shape = [self.shape[rem_self], [1], other.shape[rem_other]]
                    res_heights = [self.heights[rem_self], [1],
                                   other.heights[rem_other]]
                    # goes through the blocks
                    for h in self.heights[2]:
                        try:
                            a = self[h,1,h]
                            b = other[h,1,h]
                            tmp = np.tensordot(a,b, axes=(ind_self,ind_other))
                            res_sects[h,1,h] = tmp.reshape(a.shape[rem_self],1,
                                                           b.shape[rem_other])
                        except KeyError:
                            # One of the blocks was zero so the resulting
                            # block will be zero.
                            continue
                res = type(self)(res_shape, heights=res_heights,
                                 hmax=self.hmax, sects=res_sects,
                                 dtype=res_dtype, charge=res_charge,
                                 invar=res_invar)
        return res

    def matrix_inv(self):
        """Takes the inverse of a matrix-like SU2kMPS.

        """
        assert(self.is_mat_like())
        res_sects = {}
        # go throught the sectors
        for k,v in self.sects.items():
            sh = v.shape[0]
            cur_mat = v.reshape((sh,sh))
            res_sects[k] = np_inv(cur_mat).reshape((sh,1,sh))

        res = self.empty_like()
        res.sects = res_sects
        return res

    def matrix_trace(self):
        """Takes the trace of a matrix-like SU2kMPS.  Returns a number

        """
        assert(self.is_mat_like())
        res = 0
        # go throught the sectors
        for k,v in self.sects.items():
            sh = v.shape[0]
            cur_mat = v.reshape((sh,sh))
            res += np.trace(cur_mat)
        return res

    ##
    def contract(self, other, erase_id=True):
        """Contracts the last index of self with the first of other.
        At least one has to be 3 indices MPS are supported

        In result is:
        shape=[self.shape[:-1],[1,...,1],other.shape[1:]]
        heights=[self.heights,other.heights[1:]] 
        sects[h1,a,h,b,h2] =
        tensordot(self.sects[h1,a,h],other.sects[h,b,h2],axes=(2,0))

        if erase_id, then when either a or b are = 1, h1,a or b,h2 
        are removed, so that the result has 3 indices only.

        TODO: CHECK AND GENERALIZE

        """
        # more assertion to be added...  assert(self.is_wf() and
        # other.is_wf()) does not work since we want to use it with
        # mat_like MPS.
        assert(self.hmax == other.hmax)
        # indices to be contracted:
        i = len(self.shape)-1 #last_ind_self
        j = 0 #first_ind_other
        assert(self.compatible_indices(other, i, j))
        res_charge = 1
        res_dtype = np.result_type(self.dtype, other.dtype)
        # we do not want to handle other cases.
        res_invar = True
        
        # check case by case:
        res_sects = {}
        a = self.heights[i-1][0]
        b = other.heights[j+1][0]
        # go through all the cases if erase_ind:
        if erase_id and a == 1 and b != 1:
            # set new_sects
            for h in self.heights[i]:
                # get the possible values of heights h1 and h2
                key_1 = (h,1,h)
                h2_range = SU2k_data.fusion_range(self.hmax, h, b)
                h2_range =  [x for x in h2_range if x in other.heights[2]]
                for h2 in h2_range:
                    key_2 = (h,b,h2)
                    try:
                        u = self.sects[key_1]
                        v = other.sects[key_2]
                        tmp_block = np.tensordot(u,v,axes=(i,j))
                        new_key = key_2
                        sh = tmp_block.shape
                        sh = (sh[0],sh[2],sh[3])
                        res_sects[new_key] = np.reshape(tmp_block, sh)
                    except KeyError:
                        # One of the blocks was zero so the resulting
                        # block will be zero.
                        continue
            # set new_shape and new_heights
            res_shape = [self.shape[0]]+other.shape[j+1:]
            res_heights = other.heights
        elif erase_id and a != 1 and b == 1:
            # set new_sects
            for h in self.heights[i]:
                key_2 = (h,1,h)
                # get the possible values of heights h1 and h2
                h1_range = SU2k_data.fusion_range(self.hmax, h, a)
                h1_range =  [x for x in h1_range if x in self.heights[0]]
                for h1 in h1_range:
                    key_1 = (h1,a,h)
                    try:
                        u = self.sects[key_1]
                        v = other.sects[key_2]
                        tmp_block = np.tensordot(u,v,axes=(i,j))
                        new_key = key_1
                        sh = tmp_block.shape
                        sh = (sh[0],sh[1],sh[3])
                        res_sects[new_key] = np.reshape(tmp_block, sh)
                    except KeyError:
                        # One of the blocks was zero so the resulting
                        # block will be zero.
                        continue
            # set new_shape and new_heights
            res_shape = self.shape[:i]+[other.shape[2]]
            res_heights = self.heights
        elif erase_id and a == 1 and b == 1:
            # we do not want to allow this
            print("In SU2kMPS.contract: both a and b = 1, use matrix_dot instead")
            return 0
        else: # do not erase, if both self and other have len = 3,
              # then the first is executed and both are ok.
            if len(self.shape) == 3:
                for k,v in other.sects.items():
                    h = k[j]
                    # get the possible values of h1=self.height[0]
                    h1_range = SU2k_data.fusion_range(self.hmax, h, a)
                    h1_range =  [x for x in h1_range if x in self.heights[0]]
                    for h1 in h1_range:
                        key_1 = (h1,a,h) # key of self
                        try:
                            u = self[key_1]
                            tmp_block = np.tensordot(u,v,axes=(i,j)) 
                            new_k = key_1[:-1]+k
                            res_sects[new_k] = np.expand_dims(tmp_block, axis=i)
                        except KeyError:
                            # One of the blocks was zero so the resulting
                            # block will be zero.
                            continue
            elif len(other.shape) == 3:
                for k,v in self.sects.items():
                    h = k[i]
                    # get the possible values of h1=self.height[0]
                    h1_range = SU2k_data.fusion_range(other.hmax, h, b)
                    h1_range =  [x for x in h1_range if x in other.heights[2]]
                    for h1 in h1_range:
                        key_1 = (h,b,h1) # key of other
                        try:
                            u = other[key_1]
                            tmp_block = np.tensordot(v,u,axes=(i,j)) 
                            new_k = k+key_1[1:]
                            res_sects[new_k] = np.expand_dims(tmp_block, axis=i)
                        except KeyError:
                            # One of the blocks was zero so the resulting
                            # block will be zero.
                            continue
            else: #both self and other have length > 3:
                print("In contract: case not implemented yet, exit")
                sys.exit(1)
            # for h in self.heights[i]:
            #     # get the possible values of heights h1 and h2
            #     h1_range = SU2k_data.fusion_range(self.hmax, h, a)
            #     h1_range =  [x for x in h1_range if x in self.heights[0]]
            #     for h1 in h1_range:
            #         key_1 = (h1,a,h)
            #         h2_range = SU2k_data.fusion_range(self.hmax, h, b)
            #         h2_range = [x for x in h2_range if x in other.heights[2]]
            #         for h2 in h2_range:
            #             key_2 = (h,b,h2)
            #             try:
            #                 u = self.sects[key_1]
            #                 v = other.sects[key_2]
            #                 tmp_block = np.tensordot(u,v,axes=(i,j))
            #                 new_key = key_1[:i]+key_2
            #                 res_sects[new_key] = np.expand_dims(tmp_block, axis=i)
            #             except KeyError:
            #                 # One of the blocks was zero so the resulting
            #                 # block will be zero.
            #                 continue
            # set new_shape and new_heights: same for all cases
            shape_i = [1] * len(self.shape[i]) # replace all with 1
            res_shape = self.shape[:i]+[shape_i]+other.shape[j+1:]
            res_heights = self.heights+other.heights[j+1:]    

        # Set the new SU2kMPS and return:
        res_shape, res_heights = self.sorted_shape_heights(shape=res_shape, 
                                                           heights=res_heights)
        res = type(self)(res_shape, heights=res_heights,
                         hmax=self.hmax, sects=res_sects,
                         dtype=res_dtype, charge=res_charge,
                         invar=res_invar)
        return res

    ##
    def matrix_eig(self, chis=None, eps=0, print_errors=0, hermitian=False,
                   break_degenerate=False, degeneracy_eps=1e-6,
                   norm_type="frobenius"):
        """Find eigenvalues and eigenvectors of a matrix-like SU2kMPS. The
        input must have defval == 0, invar == True, charge == 1 and
        must be square in the sense that the dimensions must have the
        same heights and dim.

        If hermitian is True the matrix is assumed to be hermitian.

        Truncation works like for SVD.

        The output is in the form S, U, where S is a non-invariant
        vector of eigenvalues and U is a matrix that has its columns the
        eigenvectors. Both have the same dim and heights as self.

        """

        chis = self.matrix_decomp_format_chis(chis, eps)
        np_func = np.linalg.eigh if hermitian else np.linalg.eig
        assert(self.is_mat_like)
        assert(self.defval == 0)
        assert(self.invar)
        assert(self.charge == 1)
        assert(set(zip(self.heights[0], self.shape[0])) ==
               set(zip(self.heights[2], self.shape[2])))

        S_dtype = np.float_ if hermitian else np.complex_
        U_dtype = self.dtype if hermitian else np.complex_

        eigdecomps = {}
        dims = {}
        minusabs_next_eigs = [] # heap - binary tree with smallest element at node
        all_eigs = [] 
        for k,v in self.sects.items():
            # here v has shape (n,1,n) so remove the central axis:
            vshape = v.shape
            v = v.reshape((vshape[0],vshape[2]))
            if 0 not in v.shape:
                s, u = np_func(v)
                # smallest of - abs(s) first, so largest abs(s) first
                order = np.argsort(-np.abs(s))
                s = s[order]
                u = u[:,order]
                s = s.astype(S_dtype)
                u = u.astype(U_dtype)
            else:
                shp = v.shape
                m = min(shp)
                u = np.empty((shp[0], m), dtype=U_dtype)
                s = np.empty((m,), dtype=S_dtype)
            eigdecomp = (s, u)
            eigdecomps[k] = eigdecomp 
            dims[k] = 0 
            all_eigs.append(s)
            if 0 not in s.shape:
                # push to the heap, whose root has - the absolute the
                # biggest sing value, and the descending nodes contain
                # - biggest sing value for key k
                heapq.heappush(minusabs_next_eigs, (-np.abs(s[0]), k))

        try:
            # concatenate the list of ndarrays to a single ndarray
            all_eigs = np.concatenate(all_eigs)
        except ValueError:
            # all_eigs == []
            all_eigs = np.array((0,))

        
        # Truncate, if truncation dimensions are given.
        chi, dims, rel_err = type(self).find_trunc_dim(
            all_eigs, eigdecomps, minusabs_next_eigs, dims,
            chis=chis, eps=eps, break_degenerate=break_degenerate,
            degeneracy_eps=degeneracy_eps, norm_type=norm_type)

        if print_errors > 0:
            print('Relative truncation error in eig: '
                  '%.3e' % rel_err)

        # Truncate each block and create the dim for the new index.
        new_dim = []
        new_height = []
        eigdecomps = {k:v for k,v in eigdecomps.items() if dims[k] > 0}
        for k,v in eigdecomps.items():
            d = dims[k]
            if d>0:
                new_dim.append(d)
                new_height.append(k[0])
                # v[0] is the eigenvalue, v[1] the eigenvector. Retain till d
                eigdecomps[k] = (v[0][:d], v[1][:,:d])
            else:
                del(eigdecomps[k])

        # Initialize U, S.
        sh = [new_dim]
        hh = [new_height]
        sh, hh = self.sorted_shape_heights(shape=sh,heights=hh)
        S = type(self)(sh, heights=hh,
                       hmax=self.hmax, dtype=S_dtype, invar=False,
                       charge=1)
        sh = [self.shape[0], self.shape[1], new_dim]
        hh = [self.heights[0], self.heights[1], new_height]
        sh, hh = self.sorted_shape_heights(shape=sh,heights=hh)
        U = type(self)(sh, heights=hh,
                       hmax=self.hmax, dtype=U_dtype, invar=True,
                       charge=1)

        # Set the blocks of U, S and V.
        for k,v in eigdecomps.items():
            S[(k[0],)] = v[0]
            k_U = (k[0], 1, k[0])
            U[k_U] = np.expand_dims(v[1], axis=1)

        return S, U, rel_err


    ##
    def matrix_svd(self, chis=None, eps=0, print_errors=0,
                   break_degenerate=False, degeneracy_eps=1e-6,
                   norm_type="frobenius", norm_S=True, remove_small=True):
        """SVD a matrix which is encoded as an SU2kMPS with
        heights=[[h1,h2,...],[1],[h1,h2,...]] and
        shape=[[n1,n2,...],[1],[m1,m2,...]].

        This MPS must have invar == True and defval == 0.

        The optional argument chis is a list of bond dimensions. The SVD
        is truncated to one of these dimensions chi, meaning that only
        chi largest singular values are kept. If chis is a single
        integer (either within a singleton list or just as a bare
        integer) this dimension is used. If no eps==0, the largest value
        in chis is used. Otherwise the smallest chi in chis is used,
        such that the relative error made in the truncation is smaller
        than eps.

        An exception to the above is degenerate singular values. By
        default truncation is never done so that some singular values
        are included while others of the same value are left out. If
        this is about to happen chi is decreased so that none of the
        degenerate singular values is included. This default behavior
        can be changed with the keyword argument break_degenerate=True.
        The default threshold for when singular values are considered
        degenerate is 1e-6. This can be changed with the keyword
        argument degeneracy_eps.

        If print_errors > 0 truncation error is printed.

        norm_type specifies the norm used to measure the error. This
        defaults to frobenius.

        The method returns the tuple U, S, V, rel_err, where S is a
        non-invariant vector and U and V are unitary matrices. They
        are such that U.diag(S).V = self, where the equality is
        appromixate if there is truncation. More precisely, U and V
        are SU2kMPS with heights=[[h1,h2,...],[1],[h1,h2,...]] and
        shape=[[n1,n2,...],[1],[m1,m2,...]], while S has
        heights=[[h1,h2,...]] and shape [[n1,n2,...]] and is not
        invar. rel_err is the relative Frobenius norm error caused by
        the truncation.

        If norm_S=False, it returns also the norm of S.

        """
        # check that self is matrix-like and other properties:
        assert(self.is_mat_like)
        assert(self.defval == 0)
        assert(self.invar)
        chis = self.matrix_decomp_format_chis(chis, eps)

        svds = {} # dict containing S,U,V 
        # dims=dict containing the dimensions of each sector, init to 0 and
        # modified by the function find_trunc_dim
        dims = {} 
        # going to be heapq containing (-biggest_sing_value, key) for each key
        minus_next_sings = [] 
        # list contains all singular values
        all_sings = []
        for k,vv in self.sects.items():
            # here v has shape (n,1,m) so remove the central axis:
            vshape = vv.shape
            vv = vv.reshape((vshape[0],vshape[2]))
            if 0 not in vv.shape:
                # sing values in s are non-negative reals sorted in
                # descending order
                u, s, v = np.linalg.svd(vv, full_matrices=False)
            else:
                shp = vv.shape
                m = min(shp)
                u = np.empty((shp[0], m), dtype=self.dtype)
                s = np.empty((m,), dtype=np.float_)
                v = np.empty((m, shp[1]), dtype=self.dtype)
            svd = (s, u, v) # note order
            svds[k] = svd
            dims[k] = 0
            sings = svd[0]
            all_sings.append(sings)
            if 0 not in sings.shape:
                # push to the heap, whose root has - the absolute the
                # biggest sing value, and the descending nodes contain
                # - biggest sing value for key k
                heapq.heappush(minus_next_sings, (-sings[0], k))
        try:
            # flattens to a vector
            all_sings = np.concatenate(all_sings)
        except ValueError:
            # all_sings == []
            all_sings = np.array((0,))
        
        # Truncate, if truncation dimensions are given.
        chi, dims, rel_err = type(self).find_trunc_dim(
            all_sings, svds, minus_next_sings, dims,
            chis=chis, eps=eps, break_degenerate=break_degenerate,
            degeneracy_eps=degeneracy_eps, norm_type=norm_type,
            remove_small=remove_small)

        if print_errors > 0:
            print('chi',chi,'Relative truncation error in SVD: %.3e' % rel_err)

        # Truncate each block and create the dim for the new index.
        # Note that a block can disappear completely. In that case,
        # shapes and heights of U do not contain these values for the
        # right indices, but the left ones still do.  However, these
        # blocks will not be set, allowing U to be invariant.
        new_dim = []
        new_height = []
        svds = {k:v for k,v in svds.items() if dims[k] > 0}
        sum_S_sq = 0 # sum of S_i^2 over all i kept
        for k,v in svds.items():
            d = dims[k]
            if d>0:
                new_dim.append(d)
                new_height.append(k[0])
                new_S = v[0][:d]
                sum_S_sq += sum(new_S**2)
                svds[k] = (new_S, v[1][:,:d], v[2][:d,:])
            else:
                del(svds[k])

        if print_errors > 0:
            print("sum_S_sq",sum_S_sq)

        # Initialize U, S, V.
        sh = [self.shape[0], self.shape[1], new_dim]
        hh = [self.heights[0], self.heights[1], new_height]
        sh, hh = self.sorted_shape_heights(shape=sh,heights=hh)
        U = type(self)(sh, heights=hh,
                       hmax=self.hmax, dtype=self.dtype, charge=1)
        sh = [new_dim]
        hh = [new_height]
        sh, hh = self.sorted_shape_heights(shape=sh,heights=hh)
        S = type(self)(sh, heights=hh,
                       hmax=self.hmax, dtype=np.float_, invar=False,
                       charge=1)
        sh = [new_dim, self.shape[1], self.shape[2]]
        hh = [new_height, self.heights[1], self.heights[2]]
        sh, hh = self.sorted_shape_heights(shape=sh,heights=hh)
        V = type(self)(sh, heights=hh,
                       hmax=self.hmax, dtype=self.dtype, charge=self.charge)

        # Set the blocks of U, S and V.
        for k,v in svds.items():
            # this is because charge of U and S is 1
            k_U = (k[0], 1, k[0]) 
            if norm_S: 
                # normalize so that norm of S = 1 (assume frobenius norm)
                S[(k[0],)] = v[0]/np.sqrt(sum_S_sq)
            else:
                S[(k[0],)] = v[0]
            # reshape by introducing the extra dim at 1
            U[k_U] = np.expand_dims(v[1], axis=1)
            V[k] = np.expand_dims(v[2], axis=1)

        if norm_S:
            return U, S, V, rel_err
        else:
            return U, S, V, rel_err, np.sqrt(sum_S_sq)

#     def act_TL_gen(self):
#         """Act with the Temperley Lieb generator on a SU2kMPS with shape =
#         [[n1,...],[1],[1,...],[1],[n3,...]]  heights =
#         [[h1,...],[2],[h2,...],[2],[h3,...]]  to produce another
#         SU2kMPS with of the same type with sects containing the result
#         of the action of e_2

#         """
#         # checks:
#         assert(self.is_2_sites)
#         first_ind = 0
#         last_ind = 4

#         res_sects = {}
#         set_h2p = set() # set so that only unique values are retained
#         list_keys = list(self.sects.keys())
#         used_h1=set()
#         [used_h1.add(key[first_ind]) for key in list_keys]
#         used_h1 = sorted(list(used_h1))
#         for h1 in used_h1:
#             if h1 not in self.heights[last_ind]:
#                 # set only the non-zero sects, namely those for which h1=h3.
#                 # the others are automatically to defval
#                 continue
#             else:
#                 # get all possible h2' produced by the sum in the action of e_2
#                 shp1 = self.shape[first_ind][self.heights[first_ind].index(h1)]
#                 shp2 = self.shape[last_ind][self.heights[last_ind].index(h1)]
#                 range_h1_times_2 = SU2k_data.fusion_range(self.hmax, h1, 2)
#                 for h2p in range_h1_times_2:
#                     set_h2p.add(h2p)
#                     new_key = (h1,2,h2p,2,h1)
#                     # new_block=sum_{h2} sects[h1,2,h2,h1] *
#                     # weight(h2p,h2,h1) note: we do not assume that
#                     # new_key is already among the old keys so instead
#                     # of defblock we build it from scratch
#                     new_block = np.zeros((shp1,1,1,1,shp2))
#                     for h2 in range_h1_times_2:
#                         old_key = (h1,2,h2,2,h1)
#                         if old_key in self.sects.keys():
#                             tmp = self.sects[old_key]* \
#                                   SU2k_data.TL_weight(self.hmax, h1, h2, h2p)
#                             new_block += tmp
#                         # else continue
#                     res_sects[new_key] = new_block
                        
#         h2p = sorted(list(set_h2p))
#         new_dim_2 = [1] * len(h2p)
#         res_shape = self.shape[:first_ind+2]+[new_dim_2]+self.shape[last_ind-1:]
#         res_heights = self.heights[:first_ind+2]+[h2p]+self.heights[last_ind-1:]

# #        print("In SU2kMPS.act_TL_gen: shape", self.shape, " res.shape", res_shape)
# #        print("heights", self.heights, " res.heights", res_heights)
        
#         # Set the new SU2kMPS and return:
#         res_shape, res_heights = self.sorted_shape_heights(shape=res_shape, 
#                                                            heights=res_heights)
#         res = type(self)(res_shape, heights=res_heights,
#                          hmax=self.hmax, sects=res_sects,
#                          dtype=self.dtype, charge=self.charge,
#                          defval=self.defval, invar=self.invar)

#         return res

    ##
    def svd_2_sites(self, chis=None, eps=0, print_errors=0,
                   break_degenerate=False, degeneracy_eps=1e-6,
                   norm_type="frobenius", norm_S=True):
        """Compute the svd of 2 sites MPS by first fusing the left and
        indices, then flattening to a matrix-like MPS and the calling
        the matrix_svd method.

        Finally, it reshapes back, by splitting U and V to produce A
        and B.

        Returns:

        """
        assert(self.is_2_sites)

        # we need to fuse inds 0,1 and 4,3 -> fused_l_r with shape
        # [n1,...],[1],[1,...,1],[1],[m1,...] and heights
        # h,[1],h,[1],h. This can be converted to a mat-like by
        # reshaping since middle indices all have dim=1. For
        # efficiency, this can be taken into account using
        # erase_inds=True eg in first fuse. Further, since the middle
        # dims are = 1, one can split after this reshaping using the
        # original dims of self.
        
        # Fuse leftmost indices:
        fused_l = self.fuse(inds=(0,1),erase_inds=True)
#        print("fused_l",fused_l)
        # Fuse rightmost indices: 2,1 since erased indices 0,1
        fused_l_r = fused_l.fuse(inds=(2,1))
#        print("fused_l_r",fused_l_r)
#        print("Before SVD:cons",fused_l_r.check_consistency())
        # svd: 
        if norm_S == False:
            U, S, V, rel_err, sqrtS = fused_l_r.matrix_svd(chis=chis, eps=eps,
                                                print_errors=print_errors, 
                                                break_degenerate=break_degenerate,
                                                degeneracy_eps=degeneracy_eps, 
                                                norm_type=norm_type, norm_S=norm_S)
        else:
            U, S, V, rel_err = fused_l_r.matrix_svd(chis=chis, eps=eps,
                                                print_errors=print_errors, 
                                                break_degenerate=break_degenerate,
                                                degeneracy_eps=degeneracy_eps, 
                                                norm_type=norm_type, norm_S=norm_S)
        A = U.split(inds=(0,1), new_dim_i = self.shape[0], new_dim_j = self.shape[1],
                    new_h_i = self.heights[0], new_h_j = self.heights[1])
        B = V.split(inds=(2,1), new_dim_i = self.shape[4], new_dim_j = self.shape[3],
                    new_h_i = self.heights[4], new_h_j = self.heights[3])
        Lambda = S

        return A,Lambda,B,rel_err
