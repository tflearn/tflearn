from __future__ import division, print_function, absolute_import

import inspect
import os
import os.path
from inspect import getmembers, isfunction
import re
import ast

import tflearn
from tflearn import activations
from tflearn import callbacks
from tflearn import collections
import tflearn.config
from tflearn import initializations
from tflearn import metrics
from tflearn import objectives
from tflearn import optimizers
from tflearn import data_utils
from tflearn import losses
from tflearn import summaries
from tflearn import utils
from tflearn import variables
from tflearn import data_flow
from tflearn import data_preprocessing
from tflearn import data_augmentation
from tflearn.layers import conv
from tflearn.layers import core
from tflearn.layers import embedding_ops
from tflearn.layers import estimator
from tflearn.layers import merge_ops
from tflearn.layers import normalization
from tflearn.layers import recurrent
from tflearn.models import dnn, generator
from tflearn.helpers import evaluator
from tflearn.helpers import regularizer
from tflearn.helpers import summarizer
from tflearn.helpers import trainer

ROOT = 'http://tflearn.org/'

MODULES = [(activations, 'tflearn.activations'),
           (callbacks, 'tflearn.callbacks'),
           (collections, 'tflearn.collections'),
           (tflearn.config, 'tflearn.config'),
           (initializations, 'tflearn.initializations'),
           (metrics, 'tflearn.metrics'),
           (objectives, 'tflearn.objectives'),
           (optimizers, 'tflearn.optimizers'),
           (data_utils, 'tflearn.data_utils'),
           (losses, 'tflearn.losses'),
           (summaries, 'tflearn.summaries'),
           (variables, 'tflearn.variables'),
           (utils, 'tflearn.utils'),
           (data_flow, 'tflearn.data_flow'),
           (data_preprocessing, 'tflearn.data_preprocessing'),
           (data_augmentation, 'tflearn.data_augmentation'),
           (conv, 'tflearn.layers.conv'),
           (core, 'tflearn.layers.core'),
           (embedding_ops, 'tflearn.layers.embedding_ops'),
           (estimator, 'tflearn.layers.estimator'),
           (merge_ops, 'tflearn.layers.merge_ops'),
           (normalization, 'tflearn.layers.normalization'),
           (recurrent, 'tflearn.layers.recurrent'),
           (dnn, 'tflearn.models.dnn'),
           (generator, 'tflearn.models.generator'),
           (evaluator, 'tflearn.helpers.evaluator'),
           (regularizer, 'tflearn.helpers.regularizer'),
           (summarizer, 'tflearn.helpers.summarizer'),
           (trainer, 'tflearn.helpers.trainer')]

KEYWORDS = ['Input', 'Output', 'Examples', 'Arguments', 'Attributes',
            'Returns', 'Raises', 'References', 'Links', 'Yields']

SKIP = ['get_from_module', 'leakyrelu', 'RNNCell', 'resize_image']


def top_level_functions(body):
    return (f for f in body if isinstance(f, ast.FunctionDef))


def top_level_classes(body):
    return (f for f in body if isinstance(f, ast.ClassDef))


def parse_ast(filename):
    with open(filename, "rt") as file:
        return ast.parse(file.read(), filename=filename)


def format_func_doc(docstring, header):

    rev_docstring = ''

    if docstring:
        # Erase 2nd lines
        docstring = docstring.replace('\n' + '    ' * 3, '')
        docstring = docstring.replace('    ' * 2, '')
        name = docstring.split('\n')[0]
        docstring = docstring[len(name):]
        if name[-1] == '.':
            name = name[:-1]
        docstring = '\n\n' + header_style(header) + docstring
        docstring = "# " + name + docstring

        # format arguments
        for o in ['Arguments', 'Attributes']:
            if docstring.find(o + ':') > -1:
                args = docstring[docstring.find(o + ':'):].split('\n\n')[0]
                args = args.replace('    ', ' - ')
                args = re.sub(r' - ([A-Za-z0-9_]+):', r' - **\1**:', args)
                if rev_docstring == '':
                    rev_docstring = docstring[:docstring.find(o + ':')] + args
                else:
                    rev_docstring += '\n\n' + args

        for o in ['Returns', 'References', 'Links']:
            if docstring.find(o + ':') > -1:
                desc = docstring[docstring.find(o + ':'):].split('\n\n')[0]
                desc = desc.replace('\n-', '\n\n-')
                desc = desc.replace('    ', '')
                if rev_docstring == '':
                    rev_docstring = docstring[:docstring.find(o + ':')] + desc
                else:
                    rev_docstring += '\n\n' + desc

        rev_docstring = rev_docstring.replace('    ', '')
        rev_docstring = rev_docstring.replace(']\n(http', '](http')
        for keyword in KEYWORDS:
            rev_docstring = rev_docstring.replace(keyword + ':', '<h3>'
                                                  + keyword + '</h3>\n\n')
    else:
        rev_docstring = ""
    return rev_docstring


def format_method_doc(docstring, header):

    rev_docstring = ''

    if docstring:
        docstring = docstring.replace('\n' + '    ' * 4, '')
        docstring = docstring.replace('\n' + '    ' * 3, '')
        docstring = docstring.replace('    ' * 2, '')
        name = docstring.split('\n')[0]
        docstring = docstring[len(name):]
        if name[-1] == '.':
            name = name[:-1]
        docstring = '\n\n' + method_header_style(header) + docstring
        #docstring = "\n\n <h3>" + name + "</h3>" + docstring

        # format arguments
        for o in ['Arguments', 'Attributes']:
            if docstring.find(o + ':') > -1:
                args = docstring[docstring.find(o + ':'):].split('\n\n')[0]
                args = args.replace('    ', ' - ')
                args = re.sub(r' - ([A-Za-z0-9_]+):', r' - **\1**:', args)
                if rev_docstring == '':
                    rev_docstring = docstring[:docstring.find(o + ':')] + args
                else:
                    rev_docstring += '\n\n' + args

        for o in ['Returns', 'References', 'Links']:
            if docstring.find(o + ':') > -1:
                desc = docstring[docstring.find(o + ':'):].split('\n\n')[0]
                desc = desc.replace('\n-', '\n\n-')
                desc = desc.replace('    ', '')
                if rev_docstring == '':
                    rev_docstring = docstring[:docstring.find(o + ':')] + desc
                else:
                    rev_docstring += '\n\n' + desc

        rev_docstring = rev_docstring.replace('    ', '')
        rev_docstring = rev_docstring.replace(']\n(http', '](http')
        for keyword in KEYWORDS:
            rev_docstring = rev_docstring.replace(keyword + ':', '<h5>'
                                                  + keyword + '</h5>\n\n')
    else:
        rev_docstring = ""
    return rev_docstring


def classesinmodule(module):
    classes = []
    tree = parse_ast(os.path.abspath(module.__file__).replace('.pyc', '.py'))
    for c in top_level_classes(tree.body):
        classes.append(eval(module.__name__ + '.' + c.name))
    return classes


def functionsinmodule(module):
    fn = []
    tree = parse_ast(os.path.abspath(module.__file__).replace('.pyc', '.py'))
    for c in top_level_functions(tree.body):
        fn.append(eval(module.__name__ + '.' + c.name))
    return fn


def enlarge_span(str):
    return '<span style="font-size:115%">' + str + '</span>'


def header_style(header):
    name = header.split('(')[0]
    bold_name = '<span style="color:black;"><b>' + name + '</b></span>'
    header = header.replace('self, ', '').replace('(', ' (').replace(' ', '  ')
    header = header.replace(name, bold_name)
    # return '<span style="display: inline-block;margin: 6px 0;font-size: ' \
    #        '90%;line-height: 140%;background: #e7f2fa;color: #2980B9;' \
    #        'border-top: solid 3px #6ab0de;padding: 6px;position: relative;' \
    #        'font-weight:600">' + header + '</span>'
    return '<span class="extra_h1">' + header + '</span>'


def method_header_style(header):
    name = header.split('(')[0]
    bold_name = '<span style="color:black"><b>' + name + '</b></span>'
    header = header.replace('self, ', '').replace('(', ' (').replace(' ', '  ')
    header = header.replace(name, bold_name)
    return '<span class="extra_h2">' + header + '</span>'



print('Starting...')
classes_and_functions = set()


def get_func_doc(name, func):
    doc_source = ''
    if name in SKIP:
        return  ''
    if name[0] == '_':
        return ''
    if func in classes_and_functions:
        return ''
    classes_and_functions.add(func)
    header = name + inspect.formatargspec(*inspect.getargspec(func))
    docstring = format_func_doc(inspect.getdoc(func), module_name + '.' +
                                header)

    if docstring != '':
        doc_source += docstring
        doc_source += '\n\n ---------- \n\n'

    return doc_source


def get_method_doc(name, func):
    doc_source = ''
    if name in SKIP:
        return  ''
    if name[0] == '_':
        return ''
    if func in classes_and_functions:
        return ''
    classes_and_functions.add(func)
    header = name + inspect.formatargspec(*inspect.getargspec(func))
    docstring = format_method_doc(inspect.getdoc(func), header)

    if docstring != '':
        doc_source += '\n\n <span class="hr_large"></span> \n\n'
        doc_source += docstring

    return doc_source


def get_class_doc(c):
    doc_source = ''
    if c.__name__ in SKIP:
        return ''
    if c.__name__[0] == '_':
        return ''
    if c in classes_and_functions:
        return ''
    classes_and_functions.add(c)
    header = c.__name__ + inspect.formatargspec(*inspect.getargspec(
        c.__init__))
    docstring = format_func_doc(inspect.getdoc(c), module_name + '.' +
                                header)

    method_doc = ''
    if docstring != '':
        methods = inspect.getmembers(c, predicate=inspect.ismethod)
        if len(methods) > 0:
            method_doc += '\n\n<h2>Methods</h2>'
        for name, func in methods:
            method_doc += get_method_doc(name, func)
        if method_doc == '\n\n<h2>Methods</h2>':
            method_doc = ''
        doc_source += docstring + method_doc
        doc_source += '\n\n --------- \n\n'

    return doc_source

for module, module_name in MODULES:

    # Handle Classes
    md_source = ""
    for c in classesinmodule(module):
        md_source += get_class_doc(c)

    # Handle Functions
    for func in functionsinmodule(module):
        md_source += get_func_doc(func.__name__, func)

    # save module page.
    # Either insert content into existing page,
    # or create page otherwise
    path = 'templates/' + module_name.replace('.', '/')[8:] + '.md'
    if False: #os.path.exists(path):
        template = open(path).read()
        assert '{{autogenerated}}' in template, ('Template found for ' + path +
                                                 ' but missing {{autogenerated}} tag.')
        md_source = template.replace('{{autogenerated}}', md_source)
        print('...inserting autogenerated content into template:', path)
    else:
        print('...creating new page with autogenerated content:', path)
    subdir = os.path.dirname(path)
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    open(path, 'w').write(md_source)
