# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.


# Modified to meet project requirements
# -*- coding:utf-8 -*-
from . import bleu
from . import weighted_ngram_match
from . import syntax_match
from . import dataflow_match

from typing import Sequence, Tuple


def code_bleu(
    hypothesis: Sequence[str],
    pre_references: Sequence[Sequence[str]],
    lang: str,
    params: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)
    ):
    """
    Compute CodeBLEU score and intermediate scores

    Args:
        hypothesis: sequence of hypothesis strings
        pre_references: sequence consisting of references for the hypothesis
        lang (str): programming language (either 'python', 'c_sharp', or 'java')
        params (Tuple[float, float, float, float]): alpha, beta, gamma, theta parameter values 
    Returns:
        Tuple of score values (Tuple[float, float, float, float]): (ngram_match_score, weighted_ngram_match_score, syntax_match_score, dataflow_match_score, code_bleu_score)
    """
    alpha, beta, gamma, theta = params

    for i in range(len(pre_references)):
        assert len(hypothesis) == len(pre_references[i])

    references = []
    for i in range(len(hypothesis)):
        ref_for_instance = []
        for j in range(len(pre_references)):
            ref_for_instance.append(pre_references[j][i])
        references.append(ref_for_instance)
    assert len(references) == len(pre_references)*len(hypothesis)

    # calculate ngram match (BLEU)
    tokenized_hyps = [x.split() for x in hypothesis]
    tokenized_refs = [[x.split() for x in reference] for reference in references]

    ngram_match_score = bleu.corpus_bleu(tokenized_refs,tokenized_hyps)

    # calculate weighted ngram match
    keywords = [x.strip() for x in open('../evaluator/keywords/'+lang+'.txt', 'r', encoding='utf-8').readlines()]
    def make_weights(reference_tokens, key_word_list):
        return {token:1 if token in key_word_list else 0.2 \
                for token in reference_tokens}

    tokenized_refs_with_weights = [[[reference_tokens, make_weights(reference_tokens, keywords)]\
                for reference_tokens in reference] for reference in tokenized_refs]

    weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(tokenized_refs_with_weights,tokenized_hyps)

    # calculate syntax match
    syntax_match_score = syntax_match.corpus_syntax_match(references, hypothesis, lang)

    # calculate dataflow match
    dataflow_match_score = dataflow_match.corpus_dataflow_match(references, hypothesis, lang)

    code_bleu_score = alpha*ngram_match_score\
                    + beta*weighted_ngram_match_score\
                    + gamma*syntax_match_score\
                    + theta*dataflow_match_score

    return code_bleu_score





