"""Session-only source-preserving transcript experiments with one answer call."""
from __future__ import annotations
import os
from dataclasses import asdict
import streamlit as st
from . import MODEL
from .evo_grammar import digest
from .evo_public_eval import SpanAnswer,prompt
from .evo_transcript_response import restore_response
from .evo_transcript import TranscriptEncoding,compile_minimal_transcript,decode_transcript
from .evo_transcript_blocks import compile_blocks,decode_blocks,GRAMMAR_ID as BLOCKS_GRAMMAR_ID
from .evo_usage import Runner,dollars
from .tokens import count_tokens

METHODS={'Automatic (recommended)':'auto','Minimal separator':'minimal','Speaker blocks':'blocks','Raw transcript':'raw'}


def prepare(notes: str, method: str) -> TranscriptEncoding:
    if method=='auto':
        candidates=[prepare(notes,choice) for choice in ('minimal','blocks','raw')]
        return min(candidates,key=lambda candidate:candidate.encoded_tokens)
    if method=='minimal':
        encoding=compile_minimal_transcript(notes)
        decoded=decode_transcript(encoding.text,encoding.profile)
    elif method=='blocks':
        encoding=compile_blocks(notes)
        decoded=decode_blocks(encoding.text,encoding.profile)
    elif method=='raw':
        size=count_tokens(notes)
        return TranscriptEncoding(notes,'raw',size,size,0,'raw_selected','raw')
    else:
        raise ValueError('Unknown transcript method')
    if decoded!=notes or encoding.encoded_tokens>encoding.source_tokens:
        raise ValueError('Transcript integrity check failed')
    return encoding


def workspace_prompt(context: str, question: str) -> str:
    return ('Keep the response concise: at most 6 short quotations, totaling at most '
            '400 words. For broad questions, select the most relevant evidence within '
            'that limit; do not reproduce the whole transcript.\n'+prompt(context,question))


def answer_error_message(error: str) -> str:
    if 'json_invalid' in error or 'EOF while parsing' in error:
        return ('The model returned an incomplete or malformed answer. No answer was accepted. '
                'Try again, or ask a narrower question. The failed call may still have incurred usage.')
    return error


def restore_workspace_response(notes: str, encoding: TranscriptEncoding, result: dict, method: str) -> dict:
    restored=restore_response(notes,encoding,result,method)
    if not str(restored.get('error','')).startswith('Quote restoration:'):
        return restored
    # Keep independently verified quotations; never repair paraphrases into evidence.
    valid=[]
    rejected=0
    for span in result['answer']['spans']:
        single=restore_response(notes,encoding,{**result,'answer':{'found':True,'spans':[span]}},method)
        if single.get('error'):
            rejected+=1
        else:
            valid.extend(single['answer']['spans'])
    if not valid:
        return {**restored,'error':'The answer contained no verifiable quotations. Try a more specific question.'}
    return {**result,'answer':{'found':True,'spans':valid},
            'warning':f'{rejected} quotation(s) omitted because they did not exactly match the transcript. This answer may be incomplete.'}


def effective_method(method: str, encoding: TranscriptEncoding) -> str:
    if method != 'auto':
        return method
    if encoding.profile == 'raw':
        return 'raw'
    return 'blocks' if encoding.grammar_id == BLOCKS_GRAMMAR_ID else 'minimal'


def fallback_message(notes: str, method: str) -> str:
    if '\n  -' in notes:
        return 'These notes already contain compact continuation bullets. The original wording and structure will be used; no additional lossless token saving was found.'
    if method != 'auto':
        return 'This representation does not shorten these notes. Choose Automatic to try the available formats.'
    return 'Automatic checked the available lossless formats. None shortened these notes, so the original text will be used.'


def answer(notes: str, question: str, method: str, encoding: TranscriptEncoding, runner: Runner) -> dict:
    if runner.public or runner.ledger is not None:
        raise ValueError('Workspace notes require a private, memory-only runner')
    if method not in METHODS.values():
        raise ValueError('Unknown transcript method')
    selected_method = effective_method(method,encoding)
    decoded = decode_blocks(encoding.text,encoding.profile) if selected_method=='blocks' else decode_transcript(encoding.text,encoding.profile)
    if decoded!=notes:
        raise ValueError('Encoding does not match notes and selected method')
    result=runner.call(workspace_prompt(encoding.text,question),SpanAnswer,purpose='demonstration',max_output_tokens=2048)
    restored=restore_workspace_response(notes,encoding,result,selected_method)
    return {'question':question,'method':selected_method,'result':restored}


def render():
    st.subheader('Keep the meeting. Shrink the notation.')
    st.caption('Explore token savings with every utterance preserved. Answer quality may change; the Language lab shows measured tradeoffs. Notes and answers stay in this session.')
    label=st.selectbox('Transcript representation',list(METHODS),key='transcript_method_v2')
    method=METHODS[label]
    notes=st.text_area('Meeting transcript',height=220,key='transcript_notes',placeholder='Paste your meeting notes directly, including names such as Natalie Tran: or D Tran:. Automatic chooses the smallest supported representation.')
    preparation_key=digest([notes,method,MODEL,'transcript-workspace-v1'])
    if st.session_state.get('transcript_preparation_key')!=preparation_key:
        st.session_state['transcript_prepared']=prepare(notes,method)
        st.session_state['transcript_preparation_key']=preparation_key
    encoding=st.session_state['transcript_prepared']
    if notes:
        left,right,saving=st.columns(3)
        left.metric('Original context',f'{encoding.source_tokens:,} tokens')
        right.metric('Selected context',f'{encoding.encoded_tokens:,} tokens')
        saving.metric('Context reduction',f'{1-encoding.encoded_tokens/encoding.source_tokens:.1%}' if encoding.source_tokens else '0%')
        st.caption('Local context counts; actual API usage also includes the question, instruction, schema and answer. No extraction call is needed.')
        if method!='raw' and encoding.profile=='raw':
            st.info(fallback_message(notes,method))
        elif method=='auto':
            selected=effective_method(method,encoding)
            st.caption('Automatically selected: '+('Speaker blocks' if selected=='blocks' else 'Minimal separator')+'. Speaker names and all spoken text are preserved.')
        with st.expander('Inspect the representation'):
            st.code(encoding.text,language=None)
            st.caption(f'Profile: {encoding.profile} · Exact reconstruction checked · Grammar: {encoding.grammar_id}')
    question=st.text_input('Ask about the meeting',key='transcript_question')
    key=digest([notes,method,asdict(encoding),MODEL,workspace_prompt('CONTEXT','QUESTION'),SpanAnswer.model_json_schema()])
    if st.button('Answer from transcript',type='primary',key='transcript_answer'):
        if not notes.strip() or not question.strip():
            st.error('Add a transcript and question.')
        elif not os.getenv('OPENAI_API_KEY'):
            st.error('Set OPENAI_API_KEY to answer a question.')
        else:
            from openai import OpenAI
            if st.session_state.get('transcript_key')!=key:
                st.session_state['transcript_key']=key
                st.session_state['transcript_runner']=Runner(OpenAI(),public=False)
                st.session_state['transcript_answers']=[]
            with st.spinner('Reading the selected transcript…'):
                record=answer(notes,question,method,encoding,st.session_state['transcript_runner'])
                st.session_state['transcript_answers'].append(record)
    if st.session_state.get('transcript_key')==key:
        records=st.session_state.get('transcript_answers',[])
        for record in records:
            with st.chat_message('user'):
                st.write(record['question'])
            with st.chat_message('assistant'):
                result=record['result']
                if result.get('error'):
                    st.error(answer_error_message(result['error']))
                elif result['answer']['found']:
                    for span in result['answer']['spans']:
                        st.write(span)
                else:
                    st.write('The transcript does not establish an answer.')
                if result.get('warning'):
                    st.warning(result['warning'])
                usage=result.get('usage')
                if usage:
                    st.caption(f"Input {usage['input_tokens']:,} · Output {usage['output_tokens']:,} · ${dollars(usage):.6f}"+(' · session replay, no new call' if result.get('replayed') else ''))
                else:
                    st.caption('Provider usage unknown; not counted as free.')
        charged=[r['result'] for r in records if not r['result'].get('replayed')]
        if charged:
            cost=sum(dollars(r['usage']) for r in charged if r.get('usage'))
            st.caption(f'{len(charged)} new answer calls · Known cost for this transcript and format ${cost:.6f} · No extraction calls')
