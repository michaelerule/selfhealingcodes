#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function
from neurotools.system import *
import sys
__PYTHON_2__ = sys.version_info<(3, 0)
'''
Functions related to disk caching (memoization)

We need to replace hash() with hashlib since python
doesn't acutally hash by value (rather by reference)!

'''

from   collections import defaultdict

import numpy as np
import scipy.io
import inspect
import ast
import types
import os
import time, datetime
import sys
import subprocess
import warnings
import traceback
import errno
import pickle
import json
import base64
import zlib
import hashlib

from neurotools.text import wordwrap

# We will need to address this later
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

from pickle import UnpicklingError

# TODO: we should use the same pickle library as multiprocessing uses
# for better comptability with parallelism and multiprocessing
try:
    from cPickle import PicklingError
except:
    from pickle import PicklingError

import neurotools.tools
import neurotools.jobs
import neurotools.jobs.ndecorator
from   neurotools.jobs.closure   import verify_function_closure
from   neurotools.jobs.filenames import is_dangerous_filename, check_filename

from pathlib import Path

@neurotools.jobs.ndecorator.memoize
def function_hash_with_subroutines(f,force=False):
    '''
    THIS FUNCTION IS NO LONGER USED
    see `function_signature` instead!
    
    Functions may change if their subroutines change. This function computes
    a hash value that is sensitive to changes in the source code, docstring,
    argument specification, name, module, and subroutines.

    This is a recursive procedure with a fair amount of overhead.
    To allow for the possibility of mutual recursion, subroutines are
    excluded from the hash if the function has already been visited.

    This does not use the built-in hash function for functions in python.

    Is memoization possible? Making memoization compatible with graceful
    handling of potentially complex mutually recurrent call structures is
    tricky. Each function generates a call tree, which does not expand a
    node if it is already present in the call tree structure. Therefore
    there are many possible hash values for an intermediate function
    depending on how far it's call tree gets expanded, which depends on
    what has been expanded and encountered so far. Therefore, we cannot
    cache these intermediate values.

    Is it worth noting that the topology of a mutually recurrent call
    structure cannot change without changing the source code of at least
    one function in the call graph? So it suffices, to hash the subroutines,
    to expand the call graph (potentially excluding standard and system
    library functions), grab the non-recursive hash for each of these
    functions (which may be cached), and then generate the subroutine
    dependent hash by combining the non-recursive hash with the hash
    of a datastructure representing the subroutine "profile" obtained
    from the call graph.

    For now we are assuming that any decorators wrapping the function
    do not modify it's computation, and therefore can safely be stripped.
    This is an assumption and is not, in general, true.

    Note that this function cannot detect changes in effective function
    behavior that result from changes in global variables or mutable scope
    that has been closed over.
    
    Parameters
    ----------
    force : bool
        force muse be true, otherwise this function will fail with a 
        warning. 
    
    Returns
    -------
    string
        Hash of function
    '''
    if not force:
        raise NotImplementedError(
        "It is not, in general, possible to hash a function reliably")

    # repeatedly expand list of subroutines
    to_expand = {f}
    expanded  = set()
    while len(to_expand)>0:
        new_subroutines = set()
        for g in to_expand: new_subroutines |= get_subroutines(g)
        expanded |= to_expand
        to_expand = new_subroutines - expanded
    # we now have a set, we need to provide some ordering over that set
    # sort the hash values and hash that
    return hash(tuple(sorted(map(function_hash_no_subroutines,expanded))))


def get_source(f):
    '''
    Extracts and returns the source code of a function (if it exists). 
    
    I've disabled this in the self-healing codes project for...
    ... reasons.
    
    Parameters
    ----------
    f : function
        Function for which to extract source code (if possible)
    
    Returns
    -------
    string
        String containing the source code of the passed function        
    '''
    return ''
    g = neurotools.jobs.ndecorator.unwrap(f)
    try:
        return inspect.getsource(g)
    except (OSError,IOError):
        if hasattr(f,'__source__'): return f.__source__
        return inspect.getsource(f)
    raise ValueError('Cannot get function source')

@neurotools.jobs.ndecorator.memoize
def function_hash_no_subroutines(f):
    '''
    THIS FUNCTION IS NO LONGER USED
    see `function_signature` instead!
    
    See function_hash_with_subroutines. This has value is based on the

        1   Undecorated source code
        2   Docstring
        3   function name
        4   module name
        5   function argument specification

    Note that this function cannot detect changes in effective function
    behavior as a result of changes in subroutines, global variables, or
    mutable scope that has been closed over.
    
    Parameters
    ----------
    f : function
        Function for which to generate a hash value
    
    Returns
    -------
    string
        Hash value that depends on the function. Hash is constructed such
        that changes in function source code and some dependencies will
        also generate a different hash. 
    '''
    source    = get_source(f)
    docstring = ''#inspect.getdoc(f)
    name      = f.__name__
    module    = f.__module__
    argspec   = neurotools.jobs.ndecorator.sanitize(inspect.getargspec(f))
    return hash((module,name,docstring,source,argspec,subroutines))

def base64hash(obj):
    try:
        ss = ss.encode('UTF-8')
    except:
        ss = repr(obj).encode('UTF-8')
    code = base64.urlsafe_b64encode(\
        str(hashlib.sha224(ss).digest()).encode('UTF-8')).decode().replace('=','')
    return code

def base64hash2byte(obj):
    try:
        ss = ss.encode('UTF-8')
    except:
        ss = repr(obj).encode('UTF-8')
    bytes = hashlib.sha224(ss).digest()
    code = base64.urlsafe_b64encode(\
        str(bytes[:2]).encode('UTF-8')).decode().replace('=','')
    return code

__fn_sig_cache__ = dict()
def function_signature(f):
    '''
    THIS ONE IS USED
    
    Generates identifier used to locate cache corresponding to a
    particular function.

    We want to be able to cache results to dist to memoize across
    different instances and over time. However, if the code for the
    underlying function changes, we're in a pickle, as checking whether
    the change is meaningful is almost impossible.

    Caches can also become invalid if the behavior of subroutines change,
    quite tricky!

    For now, we'll check that the function module, name, argspec, source,
    and file are the same. Note that module and name identify which cache,
    and source, file, and argspec validate that the function has not
    changes significantly.

    Parameters
    ----------
    f: function

    Returns
    -------
    
    '''
    if f in __fn_sig_cache__:
        return __fn_sig_cache__[f]
    # The one thing the decorator module can't fake is where the
    # function is defined. So we can't see the source code directly if
    # we're passed a wrapped function. We can however detect this case
    # and peel away the layers to get to the underlying source. The
    # decorator module will leave the wrapped function in a variable
    # called __wrapped__, so we can follow this back to the source code
    g = f
    source    = get_source(f)
    docstring = ''#inspect.getdoc(f)
    name      = f.__name__
    module    = f.__module__
    argspec   = neurotools.jobs.ndecorator.sanitize(inspect.getargspec(f))

    identity  = (module,name)
    signature = (docstring,source,argspec)
    name = '.'.join(identity)
    #print(identity,signature)
    #code = base64.urlsafe_b64encode(\
    #    str(hash((identity,signature))&0xffff).encode('UTF-8')).decode().replace('=','')
    code = base64hash2byte((identity,signature))
    #print(name,code)
    result = name+'.'+code
    __fn_sig_cache__[f] = result
    return result
   

def signature_to_file_string(f,sig,
    mode='repr',
    compressed=True,
    base64encode=True,
    truncate=True):
    '''
    Converts an argument signature to a string if possible. 
    
    This can
    be used to store cached results in a human-readable format.
    Alternatively, we may want to simply encode the value of the
    argument signature in a string that is compatible with most file
    systems. We'd still need to perform verification on the object.

    No more than 4096 characters in path string
    No more than 255 characters in file string
    For windows compatibility try to limit it to 260 character total pathlen

    For compatibility, these characters should be avoided in paths:
        `\/<>:"|?*,@#={}'&`!%$. ASCII 0..31`

    The easiest way to avoid problematic characters without restricting the
    input is to re-encode as base 64.

    The following modes are supported.

        repr:
            Uses repr and ast.literal_eval(node_or_string) to serialize the
            argument signature. This is safe, but restricts the types permitted
            as paramteters.

        json:
            Uses json to serialize the argument signature. Argument signatures
            cannot be uniquely recovered, because tuples and lists both map to
            lists in the json representation. Restricting the types used in
            the argument signature may circumvent this.

        pickle:
            Uses pickle to serialize argument signature. This should uniquely
            store argument signatures that can be recovered, but takes more
            space. **This option no longer works in Python 3**

        human:
            Attempts a human-readable format. Experimental.

    Compression is on by defaut
    Signatures are base64 encoded by default
    '''
    sig = neurotools.jobs.ndecorator.sanitize(sig)

    if compressed and not base64encode:
        raise ValueError('Compression requires base64 encoding to be enabled')

    # A hash value gives us good distribution to control the complexity of
    # the directory tree used to manage the cache, but is not unique
    # hsh = base64.urlsafe_b64encode(str(hash(sig)&0xffff).encode('UTF-8')).decode().replace('=','')
    hsh = base64hash2byte(sig)    

    # We also need to store some information about which function this
    # is for. We'll get a human readable name identifying the funciton,
    # and a shorter hash-value to make sure we invalidate the cache if
    # the source code or function definition changes.
    fname = function_signature(f)

    # The argument spec can be mapped uniquely to a file name by converting
    # it to text, then converting this text to base64 to avoid issues with
    # special characters. Passing the text representation through zlib
    # preserves the uniqueness of the key, while reducing the overall size.
    # This improves performance
    # convert key to an encoded string
    if   mode=='repr'  : key = repr(sig)
    elif mode=='json'  : key = json.dumps(sig)
    elif mode=='pickle': key = pickle.dumps(sig)
    elif mode=='human' : key = human_encode(sig)
    else: raise ValueError('I support coding modes repr, json, and pickle\n'+
        'I don\'t recognize coding mode %s'%mode)
    # compress and base64 encode string
    key = key.encode('UTF-8')
    if compressed  : key = zlib.compress(key)
    if base64encode: key = base64.urlsafe_b64encode(key)

    # Path will be a joining of the hash and the key. The hash should give
    # good distribution, while the key means we can recover the arguments
    # from the file name.
    filename = '%s.%s.%s'%(fname,hsh,key.decode())
    # If for some reason the path is too long, complain
    if len(filename)>255:
        if truncate:
            # hash the key if it is too long and truncation is enabled
            # TODO: probably should be a better hash function?
            s  = key.decode()
            #kh = base64.urlsafe_b64encode(str(hash(s)).encode('UTF-8')).decode().replace('=','')
            kh = base64hash(s)            
            filename = '%s.%s.%s'%(fname,hsh,kh)
            filename = filename[:255]
        else:
            raise ValueError(\
                'Argument specification exceeds maximum path length.\n'+
                'Function probably accepts data as an argument,\n'+
                'rather than a key to locate data. See Joblib for a\n'+
                'caching framework that uses cryptographic hashes\n'+
                'to solve this problem. For now, we skip the cache.\n\n'+
                'The offending filename is '+filename)
    if __PYTHON_2__:
        try:
            ascii = filename.encode("utf8","ignore")
            assert unicode(ascii)==filename
            filename = ascii
        except UnicodeDecodeError:
            pass
    check_filename(filename)
    #print(filename)
    #print((fname,hsh,key.decode()))
    return filename

def file_string_to_signature(filename,mode='repr',compressed=True,base64encode=True):
    '''
    Extracts the argument key from the compressed representation in a
    cache filename entry. Inverse of signature_to_file_string.

    The following modes are supported.

    repr:
        Uses repr and ast.literal_eval(node_or_string) to serialize the
        argument signature. This is safe, but restricts the types permitted
        as paramteters.

    json:
        Uses json to serialize the argument signature. Argument signatures
        cannot be uniquely recovered, because tuples and lists both map to
        lists in the json representation. Restricting the types used in
        the argument signature may circumvent this.

    pickle:
        Uses pickle to serialize argument signature. This should uniquely
        store argument signatures that can be recovered, but takes more
        space. **This option no longer works in Python 3**

    human:
        Attempts a human-readable format. Eperimental.

    Compression is on by default
    Signatures are base64 encoded by default
    '''
    pieces = filename.split('.')
    key  = pieces[-1]
    hsh  = pieces[-2]
    name = '.'.join(pieces[:-3])

    # The argument spec can be mapped uniquely to a file name by converting
    # it to text, then converting this text to base64 to avoid issues with
    # special characters. Passing the text representation through zlib
    # preserves the uniqueness of the key, while reducing the overall size.
    # This improves performance
    if base64encode: key = base64.urlsafe_b64decode((key+'='*10).encode('UTF-8'))
    if compressed  : key = zlib.decompress(key)
    key = key.decode()
    if   mode=='repr'  : sig = ast.literal_eval(key)
    elif mode=='json'  : sig = json.loads(key)
    elif mode=='pickle': sig = pickle.loads(key)
    elif mode=='human' : sig = human_decode(key)
    else: raise ValueError('I support coding modes repr, json, and pickle\n'+
        'I don\'t recognize coding mode %s'%mode)
    sig = neurotools.jobs.ndecorator.sanitize(sig)
    return sig

def human_encode(sig):
    '''
    Formats the argument signature for saving as file name
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    sig = neurotools.jobs.ndecorator.sanitize(sig,mode='strict')
    named, vargs = sig
    if not vargs is None:
        raise ValueError(
            'Currently variable arguments are not permitted '+
            'in the human-readable format')
    result = ','.join(['%s=%s'%(k,repr(v)) for (k,v) in named])
    return result

def human_decode(key):
    '''
    Formats the argument signature for saving as file name
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    params = [k.split('=') for k in key.split(',')]
    params = tuple((n,ast.literal_eval(v)) for n,v in params)
    sig = (params,None)
    sig = neurotools.jobs.ndecorator.sanitize(sig,mode='strict')
    return sig

def get_cache_path(cache_root,f,method):
    sig = neurotools.jobs.ndecorator.argument_signature(f,args,kwargs)
    fn  = signature_to_file_string(f,sig,
            mode        ='repr',
            compressed  =True,
            base64encode=True)

    pieces   = fn.split('.')
    # first two words used as directories
    path     = cache_root + os.sep + os.sep.join(pieces[:-2]) + os.sep
    return path

def locate_cached(cache_root,f,method,args,kwargs):
    '''
    Locates the file for cached output.
    
    Parameters
    ----------
    cache_root: directory/path as string
    f: function
    methods: caching naming method
    args: function parameters
    kwargs: function keyword arguments
    
    Returns
    -------
    fn
    sig
    path
    filename
    location
    '''
    sig = neurotools.jobs.ndecorator.argument_signature(f,args,kwargs)
    fn  = signature_to_file_string(f,sig,
            mode        ='repr',
            compressed  =True,
            base64encode=True)
    pieces   = fn.split('.')
    # first two words used as directories
    path     = cache_root + os.sep + os.sep.join(pieces[:-2]) + os.sep
    # remaining pieces a filename
    filename = '.'.join(pieces[-2:])+'.'+method
    location = path+filename
    return fn,sig,path,filename,location

def validate_for_matfile(x):
    '''
    Numpy types: these should be compatible
    ==========  ================================================================================
    Type        Description
    ==========  ================================================================================
    bool\_ 	    Boolean (True or False) stored as a byte
    int8 	    Byte (-128 to 127)
    int16 	    Integer (-32768 to 32767)
    int32 	    Integer (-2147483648 to 2147483647)
    int64 	    Integer (-9223372036854775808 to 9223372036854775807)
    uint8 	    Unsigned integer (0 to 255)
    uint16 	    Unsigned integer (0 to 65535)
    uint32 	    Unsigned integer (0 to 4294967295)
    uint64 	    Unsigned integer (0 to 18446744073709551615)
    float16 	Half precision float: sign bit, 5 bits exponent, 10 bits mantissa
    float32 	Single precision float: sign bit, 8 bits exponent, 23 bits mantissa
    float64 	Double precision float: sign bit, 11 bits exponent, 52 bits mantissa
    complex64 	Complex number, represented by two 32-bit floats (real and imaginary components)
    complex128 	Complex number, represented by two 64-bit floats (real and imaginary components)
    ==========  ================================================================================
    
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    safe = (np.bool_  , np.int8     , np.int16 , np.int32 , np.int64  ,
                  np.uint8  , np.uint16   , np.uint32, np.uint64, np.float32,
                  np.float64, np.complex64, np.complex128)
    if not type(x) == np.ndarray: x = np.array(x)
    if len(shape(x))<2:
        raise ValueError("One-dimensional arrays cannot be stored safely in matfiles")
    if x.dtype == np.object:
        # object arrays will be converted to cell arrays,
        # we need to make sure each cell can be stored safely
        return map(validate_for_matfile,x)
    if not x.dtype in safe:
        raise ValueError("Numpy type %s is not on the list of compatible types"%x.dtype)
    return True


def validate_for_numpy(x):
    '''
    Check whether an array-like object can safely be stored in a numpy
    archive. 
    
    Numpy types: these should be compatible
    ==========  ================================================================================
    Type        Description
    ==========  ================================================================================
    bool\_ 	    Boolean (True or False) stored as a byte
    int8 	    Byte (-128 to 127)
    int16 	    Integer (-32768 to 32767)
    int32 	    Integer (-2147483648 to 2147483647)
    int64 	    Integer (-9223372036854775808 to 9223372036854775807)
    uint8 	    Unsigned integer (0 to 255)
    uint16 	    Unsigned integer (0 to 65535)
    uint32 	    Unsigned integer (0 to 4294967295)
    uint64 	    Unsigned integer (0 to 18446744073709551615)
    float16 	Half precision float: sign bit, 5 bits exponent, 10 bits mantissa
    float32 	Single precision float: sign bit, 8 bits exponent, 23 bits mantissa
    float64 	Double precision float: sign bit, 11 bits exponent, 52 bits mantissa
    complex64 	Complex number, represented by two 32-bit floats (real and imaginary components)
    complex128 	Complex number, represented by two 64-bit floats (real and imaginary components)
    ==========  ================================================================================
    
    
    Parameters
    ----------
    x : object
        array-like object; 
    
    Returns
    -------
    bool
        True if the data in `x` can be safely stored in a Numpy archive
    '''
    safe = (np.bool_  , np.int8     , np.int16 , np.int32 , np.int64  ,
                  np.uint8  , np.uint16   , np.uint32, np.uint64, np.float32,
                  np.float64, np.complex64, np.complex128)
    if not type(x) == np.ndarray:
        x = np.array(x)
    if x.dtype == np.object:
        # object arrays will be converted to cell arrays,
        # we need to make sure each cell can be stored safely
        try:
            ix = iter(x)
        except TypeError as te:
            raise ValueError('is not iterable')
        return map(validate_for_numpy,x)
    if not x.dtype in safe:
        raise ValueError("Numpy type %s is not on the list of compatible types"%x.dtype)
    return True

def read_cache_entry(location,method):
    if method=='pickle':
        with open(location,'rb') as openfile:
            return pickle.load(openfile)
    elif method =='mat':
        return scipy.io.loadmat(location)['varargout']
    elif method =='npy':
        return np.load(location,allow_pickle=True)

def disk_cacher(
    cache_location,
    method     = 'npy',
    write_back = True,
    skip_fast  = False,
    verbose    = False,
    allow_mutable_bindings=False,
    CACHE_IDENTIFIER='.__neurotools_cache__'):
    '''
    Decorator to memoize functions to disk.
    Currying pattern here where cache_location creates decotrators

    write_back:

         True: Default. Computed results are saved to disk

        False: Computed results are not saved to disk. In this case of
               Hierarchical caches mapped to the filesystem, a background
               rsync loop can handle asynchronous write-back.

    method:

         p: Use pickle to store cache. Can serialize all objects but
            seriously slow! May not get ANY speedup due to time costs if
            pickling and disk IO

       mat: Use scipy.io.savemat and scipy.io.loadmat. Nice because it's
            compatible with matlab. Unfortunately, can only store numpy types
            and data that can be converted to numpy types. Data conversion
            may alter the type signature of the return arguments when
            retrieved from the cache.

       npy: Use built in numpy.save functionality. Experimental. Will
            likely only work if the return value is a single numpy array?

      hdf5: Not supported. Will be implemented in the future
      
      
    
    Parameters
    ----------
    cache_location : string
        Path to disk cache
    method : string, default 'npy',
        Storange format for caches. Can be 'pickle', 'mat' or 'npy'
    write_back : bool, default=True
        Whether to copy new cache value back to the disk cache. If false,
        then previously cached values can be read but new entries will not
        be creates
    skip_fast : bool, default=False
        Attempt to simply re-compute values which are taking too long to
        retrieve from the cache. Experimental, should not be used.
    verbose : bool, default=False
        Whether to print detailde logging information
    allow_mutable_bindings : bool, default=False
        Whether to allow caching of functions that close over mutable
        scope. Such functions are more likely to return different results
        for the same arguments, leading to invalid cached return values.
    CACHE_IDENTIFIER : string, default='.__neurotools_cache__'
        subdirectory name for disk cache.
    
    Returns
    -------
    cached : disk cacher object
        TODO
    '''
    VALID_METHODS = ('pickle','mat','npy')
    assert method in VALID_METHODS
    
    cache_location = os.path.abspath(cache_location)+os.sep
    cache_root     = cache_location+CACHE_IDENTIFIER
    neurotools.tools.ensure_dir(cache_location)
    neurotools.tools.ensure_dir(cache_root)

    if verbose:
        prefix = '\x1b[48;5;8m\x1b[38;5;195m'
        suffix = '\x1b[39m\x1b[49m\x1b[22m\x1b[23m\x1b[24m\x1b[25m\x1b[27m\x1b[28m\x1b[29m\x1b[23m\x1b[22m'
        def log(*args):           
            def subroutine(args,flush=True,force_ascii=False):
                lines   = [l for a in args for l in str(a).split('\n')]
                message = [('CACHE: '+lines[0]).ljust(80)] + \
                          [('  '+l).ljust(80) for l in lines[1:]]
                if force_ascii:
                    lines = [l.encode("ascii","replace").decode('ascii') for l in lines]
                lines   = ' \n'.join([prefix+l+suffix for l in message])
                print(lines,flush=flush)
            flush       = True
            force_ascii = False
            for retry in range(3):
                try:
                    subroutine(args,flush,force_ascii)
                    return
                except UnicodeEncodeError:
                    force_ascii=True
                except BlockingIOError:
                    flush = False
    else:
        def log(*args):
            pass

    def cached(f):
        '''
        This is a wrapper for memoizing results to disk. 
        This docstring should be overwritten by the docstring of
        the wrapped function.
        '''
        if not allow_mutable_bindings:
            verify_function_closure(f)
            
        # Patch for 2/3 compatibility
        if __PYTHON_2__:
            FileError = IOError
        else:
            FileError = FileNotFoundError
            
        @neurotools.jobs.ndecorator.robust_decorator
        def wrapped(f,*args,**kwargs):
            '''
            This is a wrapper for memoizing results to disk. 
            This docstring should be overwritten by the docstring of
            the wrapped function.
            '''
            t0 = neurotools.tools.current_milli_time()

            # Store parameters; I hope we can save these in numpy...
            params = (args,tuple(list(kwargs.items())))

            try:
                fn,sig,path,filename,location = locate_cached(cache_root,f,method,args,kwargs)
            except ValueError as exc:
                sys.stderr.write('Generating cache key failed\n')
                traceback.print_exc()
                time,result = f(*args,**kwargs)
                return result
            
            result = None
            if os.path.isfile(location):
                try:
                    result = read_cache_entry(location,method)
                    log(wordwrap('Retrieved  %s'%path,70,'/'),
                        '%s.%s'%(f.__module__,f.__name__),
                        wordwrap('%s'%neurotools.jobs.ndecorator.print_signature(sig),70,','))
                except (ValueError, EOFError, OSError, IOError, FileError, BlockingIOError, UnpicklingError) as exc:
                    if verbose: log('  File reading failed')

            if not result is None:
                params,result = result
            else:
                #wordwrap(text,width=80,sep=' ')
                log(wordwrap('Computing %s'%cache_location,70,'/'),
                    '%s.%s'%(f.__module__,f.__name__),
                    wordwrap('%s'%neurotools.jobs.ndecorator.print_signature(sig),70,','))

                # Evaluate function
                _t0 = neurotools.tools.current_milli_time()
                result = f(*args,**kwargs)
                _t1 = neurotools.tools.current_milli_time()
                compute_cost = float(_t1-_t0)
                log(wordwrap('%s'%path,70,'/'),'Took %d ms'%compute_cost)

                # Save Cached output to disk
                if write_back:
                    savedata = (params,result)
                    neurotools.tools.ensure_dir(path)
                    Path(location).touch()
                    log(wordwrap('Writing %s'%path,70,'/'))
                    try:
                        if method=='pickle':
                            with open(location,'wb') as openfile:
                                pickle.dump(savedata,openfile,protocol=pickle.HIGHEST_PROTOCOL)
                        elif method =='mat':
                            validated_result = validate_for_matfile(savedata)
                            if validated_result is None:
                                raise ValueError('Error: return value cannot be safely packaged in a matfile')
                            scipy.io.savemat(location,{'varargout':savedata})
                        elif method =='npy':
                            validated_result = validate_for_numpy(savedata)
                            if validated_result is None:
                                raise ValueError('Error: return value cannot be safely packaged in a numpy file')
                            np.save(location, savedata)
                    except (ValueError, EOFError, OSError, IOError, FileError, BlockingIOError, PicklingError) as exc2:
                        log(wordwrap('Saving %s FAILED'%cache_location,70,'/'),
                            '%s.%s'%(f.__module__,f.__name__),
                            wordwrap('%s'%neurotools.jobs.ndecorator.print_signature(sig),70,','),
                            traceback.format_exc())

                    log(wordwrap('Wrote %s'%path,70,'/'),
                        'For function %s.%s'%(f.__module__,f.__name__),
                        wordwrap('Argument signature %s'%neurotools.jobs.ndecorator.print_signature(sig),70,','))
                    if verbose:
                        try:
                            st        = os.stat(location)
                            du        = st.st_blocks * st.st_blksize
                            _t2       = neurotools.tools.current_milli_time()
                            disk_cost = float(_t2-_t1)
                            bytes_per_ms_disk      = float(du)/(1+disk_cost   )
                            bytes_per_ms_recompute = float(du)/(1+compute_cost)
                            boost     = bytes_per_ms_disk - bytes_per_ms_recompute
                            quality   = boost/(1+float(du))
                            log('Size on disk is %d'%du,
                                'Recompute code %d ms'%compute_cost,
                                'IO overhead    %d ms'%disk_cost,
                                'Bytes per ms, disk    %0.4f'%bytes_per_ms_disk,
                                'Bytes per ms, compute %0.4f'%bytes_per_ms_recompute,
                                'Expected boost        %0.4f'%boost,
                                'Time-space quality    %0.4f'%quality)
                        except (OSError) as exc3:
                            log(traceback.format_exc())
                    # Skipping when the cache is slower than recompute is not yet supported
                    # if skip_fast and boost<0:
                    #    if verbose:
                    #        print('  WARNING DISK IO MORE EXPENSIVE THAN RECOMPUTING!')
                    #        print('  We should really do something about this?')
                    #        print('  Zeroing out the file, hopefully that causes it to crash on load?')
                    #    with open(location, 'w'): pass
            return result
        def purge(*args,**kwargs):
            '''
            Delete cache entries matching arguments. This is a destructive
            operation, execute with care.
    
            Parameters
            ----------
            *args
                Arguments forward to the `locate_cached` function. Matching
                cache entries will be deleted.
            **kwargs
                Keyword arguments forward to the `locate_cached` function
                Matching cache entries will be deleted.
            '''
            for method in VALID_METHODS:
                fn,sig,path,filename,location = locate_cached(cache_root,f,method,args,kwargs)
                print('Deleting %s'%location)
                try:
                    os.remove(location)
                    print('Deleted %s'%location)
                except OSError as ee:
                    if ee.errno==2:
                        print('%s does not exist'%location)
                    else:
                        raise
            pass
        def lscache(verbose=False):
            path = cache_root + os.sep + os.sep.join(function_signature(f).split('.'))
            try:
                files = os.listdir(path)
            except:
                files = []
            if verbose:
                log('Cache %s contains:'%path)
                log('\n  '+'\n  '.join([f[:20]+'…' for f in files]))
            return path,files
        decorated            = wrapped(f)
        decorated.purge      = purge
        decorated.cache_root = cache_root
        decorated.lscache    = lscache 
        return decorated
    return cached


def hierarchical_cacher(fast_to_slow,
        method='npy',
        write_back=True,
        verbose=False,
        allow_mutable_bindings=False,
        CACHE_IDENTIFIER ='.__neurotools_cache__'):
    '''
    Construct a filesystem cache defined in terms of a hierarchy from
    faster to slower (fallback) caches.
    
    Parameters
    ----------
    fast_to_slow : tuple of strings
        list of filesystem paths for disk caches in order from the fast
        (default or main) cache to slower.
        
    Other Parameters
    ----------------
    method: string, default 'npy'
        cache storing method;
    write_back : bool, default True
        whether to automatically copy newly computed cache values to 
        the slower caches
    verbose : bool, defaults to `False`
        whether to print detailed logging iformation to standard out
        when manipulating the cache
    allow_mutable_bindings : bool, default False
        If true, then "unsafe" namespace bindings, for example user-
        defined functions, will be allowed in disk cached functions. 
        If a cached function calls subroutines, and those subroutines
        change, the disk cacher cannot detect the implementation different.
        Consequentially, it cannot tell whether old cached values are 
        invalid. 
    CACHE_IDENTIFIER : str, default '.__neurotools_cache__'
        (sub)folder name to store cached results
    
    Returns
    -------
    hierarchical : decorator
        A hierarchical disk-caching decorator that can be used to memoize
        functions to the specified disk caching hierarchy. 
    '''
    slow_to_fast = fast_to_slow[::-1] # reverse it
    all_cachers  = []
    def hierarchical(f):
        # disable write-back on the slow caches
        for location in slow_to_fast[:-1]:
            f = disk_cacher(location,
                method                 = method,
                write_back             = write_back,
                verbose                = verbose,
                allow_mutable_bindings = allow_mutable_bindings,
                CACHE_IDENTIFIER       = CACHE_IDENTIFIER)(f)
            all_cachers.append(f)
        # use write-back only on the fast cache
        location = slow_to_fast[-1]
        f = neurotools.jobs.cache.disk_cacher(location,
            method                 = method,
            write_back             = True,
            verbose                = verbose,
            allow_mutable_bindings = allow_mutable_bindings,
            CACHE_IDENTIFIER       = CACHE_IDENTIFIER)(f)
        def purge(*args,**kwargs):
            '''
            Purge each of the constituent cachers
            '''
            for cacher in all_cachers:
                if hasattr(cacher,'purge'):
                    cacher.purge(*args,**kwargs)
        f.purge = purge
        return f
    return hierarchical




