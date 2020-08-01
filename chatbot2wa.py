from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import os
import pickle
import copy
import sys
import html

from utils import TextLoader
from model import Model

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome import options
import time

from gtts import gTTS
from playsound import playsound

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='models/reddit',
                       help='model directory to store checkpointed models')
    parser.add_argument('-n', type=int, default=500,
                       help='number of characters to sample')
    parser.add_argument('--prime', type=str, default=' ',
                       help='prime text')
    parser.add_argument('--beam_width', type=int, default=2,
                       help='Width of the beam for beam search, default 2')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='sampling temperature'
                       '(lower is more conservative, default is 1.0, which is neutral)')
    parser.add_argument('--topn', type=int, default=-1,
                        help='at each step, choose from only this many most likely characters;'
                        'set to <0 to disable top-n filtering.')
    parser.add_argument('--relevance', type=float, default=-1.,
                       help='amount of "relevance masking/MMI (disabled by default):"'
                       'higher is more pressure, 0.4 is probably as high as it can go without'
                       'noticeably degrading coherence;'
                       'set to <0 to disable relevance masking')
    args = parser.parse_args()
    sample_main(args)

def get_paths(input_path):
    if os.path.isfile(input_path):
        # Passed a model rather than a checkpoint directory
        model_path = input_path
        save_dir = os.path.dirname(model_path)
    elif os.path.exists(input_path):
        # Passed a checkpoint directory
        save_dir = input_path
        checkpoint = tf.train.get_checkpoint_state(save_dir)
        if checkpoint:
            model_path = checkpoint.model_checkpoint_path
        else:
            raise ValueError('Checkpoint not found in {}.'.format(save_dir))
    else:
        raise ValueError('save_dir is not a valid path.')
    return model_path, os.path.join(save_dir, 'config.pkl'), os.path.join(save_dir, 'chars_vocab.pkl')

def sample_main(args):
    model_path, config_path, vocab_path = get_paths(args.save_dir)
    # Arguments passed to sample.py direct us to a saved model.
    # Load the separate arguments by which that model was previously trained.
    # That's saved_args. Use those to load the model.
    with open(config_path, 'rb') as f:
        saved_args = pickle.load(f)
    # Separately load chars and vocab from the save directory.
    with open(vocab_path, 'rb') as f:
        chars, vocab = pickle.load(f)
    # Create the model from the saved arguments, in inference mode.
    print("Creating model...")
    saved_args.batch_size = args.beam_width
    net = Model(saved_args, True)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # Make tensorflow less verbose; filter out info (1+) and warnings (2+) but not errors (3).
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(net.save_variables_list())
        # Restore the saved variables, replacing the initialized values.
        print("Restoring weights...")
        saver.restore(sess, model_path)
        chatbot(net, sess, chars, vocab, args.n, args.beam_width,
                args.relevance, args.temperature, args.topn)

def initial_state(net, sess):
    # Return freshly initialized model states.
    return sess.run(net.zero_state)

def forward_text(net, sess, states, relevance, vocab, prime_text=None):
    if prime_text is not None:
        for char in prime_text:
            if relevance > 0.:
                # Automatically forward the primary net.
                _, states[0] = net.forward_model(sess, states[0], vocab[char])
                # If the token is newline, reset the mask net state; else, forward it.
                if vocab[char] == '\n':
                    states[1] = initial_state(net, sess)
                else:
                    _, states[1] = net.forward_model(sess, states[1], vocab[char])
            else:
                _, states = net.forward_model(sess, states, vocab[char])
    return states

def sanitize_text(vocab, text): # Strip out characters that are not part of the net's vocab.
    return ''.join(i for i in text if i in vocab)

def initial_state_with_relevance_masking(net, sess, relevance):
    if relevance <= 0.: return initial_state(net, sess)
    else: return [initial_state(net, sess), initial_state(net, sess)]

def possibly_escaped_char(raw_chars):
    if raw_chars[-1] == ';':
        for i, c in enumerate(reversed(raw_chars[:-1])):
            if c == ';' or i > 8:
                return raw_chars[-1]
            elif c == '&':
                escape_seq = "".join(raw_chars[-(i + 2):])
                new_seq = html.unescape(escape_seq)
                backspace_seq = "".join(['\b'] * (len(escape_seq)-1))
                diff_length = len(escape_seq) - len(new_seq) - 1
                return backspace_seq + new_seq + "".join([' '] * diff_length) + "".join(['\b'] * diff_length)
    return raw_chars[-1]
def clean(t):
    t2=""
    f=1
    prev = ">"
    for x in t:
       if x=="<":
           f=2
       elif x==">":
           f=1
       elif f==1:
           if prev == ">":
               t2+=" "
           t2 += x
       prev = x
    t2 = t2.replace("&amp;","&")
    t2 = t2.replace("&nbsp;"," ")
    t2 = t2.replace("&gt;",">")
    t2 = t2.replace("&lt;","<")
    return t2+"\n"
def replaceadd(str):
    t = ""
    for x in str:
        if x == '+':
            t += "%2b"
        else:
            t += x
    return t
def selectedoption(str):
    str = str.split('selected="1"')[1]
    str = str.split('>')[1]
    str = str.split('<',1)[0]
    return str
def blacklist(str):
    black_list = ["fuck",
                  "damn",
                  "pussy",
                  "idiot",
                  "ass",
                  "stupid",
                  "cunt",
                  "vagina",
                  "crap",
                  "piss",
                  "bitch",
                  "bloody",
                  "darn",
                  "douche",
                  "bollocks",
                  "fag",
                  "dick",
                  "bastard",
                  "slut",
                  "cock",
                  "shit",
                  "arsehole",
                  "bugger",
                  "tits",
                  "clit",
                  "sucker",
                  "twat",
                  "whore",
                  "prick"
                  ]
    str = str.lower()
    for word in black_list:
        if word in str:
            return True
    return False
def assist(user_input,driver,wadriver,input_box):
    command = user_input.split(' ')
    reply = "sorry could not process the assistance"
    if command[1] == "time":
        try:
            driver.get("https://www.google.co.in/search?q=" + user_input.split('#', 1)[1])
            vartime = driver.find_elements_by_class_name("vk_bk")
            input_box.send_keys("> "+vartime[0].get_attribute('innerHTML') + Keys.ENTER)
        except:
            input_box.send_keys("> could not tell the time at this moment."+Keys.ENTER)
    elif command[1] == "define":
            driver.get("https://www.google.co.in/search?q=" + user_input.split('#', 1)[1])
            word = driver.find_elements_by_class_name("vk_ans")
            if word.__len__() != 0:
                input_box.send_keys("> Defining " + clean(word[0].get_attribute('innerHTML')) + '.' + Keys.ENTER)
            meaning = driver.find_elements_by_class_name("PNlCoe")
            if meaning.__len__() == 0:
                meaning = driver.find_elements_by_class_name("CLPzrc")
                if meaning.__len__() == 0:
                    input_box.send_keys("> could not find a meaning for " + user_input.split(' ', 2)[2] + Keys.ENTER)
            for x in meaning:
                t = x.get_attribute('innerHTML')
                t = clean(t)
                input_box.send_keys("> " + t + Keys.ENTER)
    elif command[1] == "how":
        try:
            flag = True
            driver.get("https://www.google.co.in/search?q=" + user_input.split(' ', 1)[1])
            db = driver.find_elements_by_class_name("mod")
            d = 1
            for data in db:
                if d == 1:
                    d = 2
                    continue
                steps = clean(data.get_attribute("innerHTML"))
                print("["+steps+"]")
                if (steps[-4] == '2' and steps[-3] == '0'):
                    steps = steps[0:-12]
                if (steps[-5] == '2' and steps[-4] == '0'):
                    steps = steps[0:-13]
                if (steps.__contains__("More items...")):
                    steps = steps[0:-14]
                steps = steps.replace("...", " ")
                input_box.send_keys(steps +Keys.ENTER)
                flag = False
            if flag:
                input_box.send_keys("> count not find ans to : " + user_input.split(' ', 1)[1] + Keys.ENTER)
            else:
                input_box.send_keys("> end." + Keys.ENTER)
        except:
            input_box.send_keys("> count not find ans to : " + user_input.split(' ', 1)[1] + Keys.ENTER)
    elif command[1] == "movies":
        flag = True
        try:
            driver.get("https://www.google.co.in/search?q=" + user_input.split(' ', 1)[1])
            movi = driver.find_elements_by_class_name("kltat")
            count =1
            for m in movi:
                if count > 30:
                    break
                count+=1
                t = m.get_attribute("innerHTML")
                t = clean(t)
                input_box.send_keys("> " + t+Keys.ENTER)
                flag = False
        except:
            input_box.send_keys(""+Keys.ENTER)
        if flag:
            input_box.send_keys("> could not list the movies."+Keys.ENTER)
    elif command[1] == "date":
        try:
            driver.get("https://www.google.co.in/search?q=" + user_input.split(' ', 1)[1])
            dt = driver.find_elements_by_class_name("vk_bk")
            t = dt[0].get_attribute("innerHTML")
            t = clean(t)
            input_box.send_keys("> " + t+Keys.ENTER)
        except:
            input_box.send_keys("> could not tell the date at this moment."+Keys.ENTER)
    elif command[1] == "weather":
        try:
            driver.get("https://www.google.co.in/search?q=" + user_input.split(' ', 1)[1])
            dates = driver.find_elements_by_class_name("wob_df")
            flag = True
            for date in dates:
                date.click()
                dt = date.get_attribute("innerHTML")
                dt = dt.split('aria-label="', 1)[1]
                input_box.send_keys("> " + dt.split('"', 1)[0]+Keys.SHIFT, Keys.ENTER)
                flag = False
                dt = dt.split('alt="')[1]
                input_box.send_keys("> " + dt.split('"')[0]+Keys.SHIFT, Keys.ENTER)
                dt = dt.split(':inline">')
                input_box.send_keys(">  Temperature : " + dt[1].split("<", 1)[0] + " - " + dt[2].split("<", 1)[0] + " degree Celsius"+Keys.SHIFT, Keys.ENTER)
                details = driver.find_elements_by_class_name("wob-dtl")
                for d in details:
                    t = d.get_attribute("innerHTML")
                    t = clean(t)
                    input_box.send_keys("> " + t.split('%')[0] + '%'+Keys.SHIFT, Keys.ENTER)
                    input_box.send_keys("> " + t.split('%')[1] + '%'+Keys.SHIFT, Keys.ENTER)
                    t = t.split('%')[2]
                    input_box.send_keys("> " + t.split('km/h')[0] + 'km/h'+Keys.ENTER)
                    flag = False
            if flag:
                input_box.send_keys("> some error occured. query could not be processed."+Keys.ENTER)
        except:
            input_box.send_keys("> could not tell the weather at this moment."+Keys.ENTER)
    elif command[1] == "news":
        flag = True
        try:
            topic = user_input.split('news')[1]
            driver.get(
                "https://news.google.com/news/search/section/q/" + topic + "/" + topic + "?hl=en-IN&gl=IN&ned=in")
            news = driver.find_elements_by_class_name("DY5T1d")
            c = 1
            for n in news:
                try:
                    if c > 20:
                        break
                    c += 1
                    headline = clean(n.get_attribute("innerHTML"))
                    print("Link = " + n.get_attribute("href"))
                    input_box.send_keys(headline +" ( link : "+ n.get_attribute("href") + " )" +Keys.ENTER)
                    flag =False
                except:
                    1
        except:
            input_box.send_keys("> error"+Keys.ENTER)
        if flag:
            input_box.send_keys("> no news could be found at this moment."+Keys.ENTER)
        else:
            input_box.send_keys("> end" + Keys.ENTER)
    elif command[1] == "curr":
        try:
            driver.get("https://www.google.co.in/search?q=convert " + user_input.split(' ', 2)[2])
            input_box.send_keys("> " + driver.find_element_by_id("knowledge-currency__src-input").get_attribute(
                "value") + " " + selectedoption(
                driver.find_element_by_id("knowledge-currency__src-selector").get_attribute(
                    "innerHTML")) + " = " + driver.find_element_by_id("knowledge-currency__tgt-input").get_attribute(
                "value") + " " + selectedoption(
                driver.find_element_by_id("knowledge-currency__tgt-selector").get_attribute("innerHTML"))+Keys.ENTER)
        except:
            input_box.send_keys("> could not convert the currency."+Keys.ENTER)
    elif command[1] == "calc":
        try:
            ques = user_input.split('calc')[1]
            ques = replaceadd(ques)
            driver.get("https://www.google.co.in/search?q=" + ques)
            input_box.send_keys("> calculating..." + Keys.ENTER)
            ans = driver.find_element_by_id('cwos').get_attribute("innerHTML")
            input_box.send_keys("> " + driver.find_element_by_class_name("vUGUtc").get_attribute("innerHTML").replace("&nbsp;","") + ans + Keys.ENTER)
        except:
            input_box.send_keys("> could not calculate the given expression"+Keys.ENTER)
    elif command[1] == "pincode":
        try:
            place = user_input.split(' ', 2)[2]
            place = place.replace(' ', '-')
            driver.get("http://www.pincodebox.in/" + place + ".html")
            links = driver.find_elements_by_tag_name("a")
            input_box.send_keys("> "+links[1].get_attribute("href").split("/")[-1]+Keys.ENTER)
        except:
            input_box.send_keys("> could not tell the pincode of this place."+Keys.ENTER)
    elif command[1] == "loc":
        place = user_input.split('loc ', 1)[1]
        placel = place.replace(' ', '+')
        print(placel)
        # driver.get("https://www.google.co.in/search?q=" + placel + "&npsic=0&rflfq=1&rlha=0&rllag=28581647,77059441,2497&tbm=lcl&ved=0ahUKEwiqnprXr9XaAhVLG5QKHSObDtIQjGoIYw&tbs=lrf:!2m1!1e2!2m1!1e3!3sIAE,lf:1,lf_ui:2&rldoc=1#rlfi=hd:;si:;mv:!1m3!1d27587.311026682404!2d77.0556183!3d28.59525525!2m3!1f0!2f0!3f0!3m2!1i335!2i378!4f13.1;tbs:lrf:!2m1!1e2!2m1!1e3!3sIAE,lf:1,lf_ui:2")
        driver.get("https://www.google.com/search?hl=en-IN&authuser=0&rlz=1C5CHFA_enIN845IN846&q=" + placel + "&npsic=0&rflfq=1&rlha=0&rllag=28626246,77083473,957&tbm=lcl&ved=2ahUKEwje-Pvcse3hAhWjo1kKHa5QAVcQjGp6BAgKEDo&tbs=lrf:!2m1!1e2!2m1!1e3!3sIAE,lf:1,lf_ui:2&rldoc=1#rlfi=hd:;si:;mv:!1m2!1d28.6352625!2d77.108156!2m2!1d28.6065282!2d77.0705324;tbs:lrf:!2m1!1e2!2m1!1e3!3sIAE,lf:1,lf_ui:2")
        driver.find_element_by_id("lst-ib").send_keys(Keys.CONTROL, "a", place + Keys.ENTER)
        time.sleep(1)
        locdata = driver.find_elements_by_class_name("VkpGBb")
        flag = True
        count = 1
        for data in locdata:
            if count >8:
                break
            count+=1
            try:
                str = data.get_attribute("innerHTML")
                loc = ""
                str = clean(str)
                str = str.replace('Website', "")
                str = str.replace('Directions', "")
                str = str.replace("&nbsp;", "")
                '''
                try:
                    str = str.split(" km")[0]+str.split(" ",-1)[0]
                except:
                    str = str
                '''
                input_box.send_keys(str+Keys.ENTER)
                data.click()
                time.sleep(1)
                try:
                    input_box.send_keys("ADD: " + clean(driver.find_element_by_class_name("LrzXr").get_attribute("innerHTML"))+Keys.ENTER)
                except:
                    1
                try:
                    input_box.send_keys("PHONE: " + clean(driver.find_element_by_class_name("kno-fv").get_attribute("innerHTML"))+Keys.ENTER)
                except:
                    1
                '''
                try:
                    str = str.split("&nbsp")[0]+str.split("&nbsp")[1]
                except:
                    str = str
                '''
                loc = driver.find_elements_by_class_name("CL9Uqc")
                for locs in loc:
                    try:
                        print("map: " + locs.get_attribute("data-url"))
                        input_box.send_keys("map: google.com" + locs.get_attribute("data-url")+Keys.ENTER)
                    except:
                        1
                flag = False
            except:
                continue
        if flag:
            input_box.send_keys("> could not find any such location at this moment."+Keys.ENTER)
        else:
            input_box.send_keys("> end"+Keys.ENTER)
    elif command[1] == "horoscope":
        flag = False
        try:
            sign = user_input.split("horoscope ")[1]
            sunsign = ['capricorn', 'aquarius', 'pisces', 'aries', 'taurus', 'gemini', 'cancer', 'leo', 'virgo',
                       'libra', 'scorpio', 'sagittarius']
            sign = sunsign[int(sign) - 1]
            driver.get("http://astrostyle.com/daily-horoscopes/" + sign + "-daily-horoscope/")
            horo = driver.find_elements_by_class_name("weekday_div")
            for x in horo:
                try:
                    str = clean(x.get_attribute("innerHTML"))
                    input_box.send_keys("> " + str+Keys.ENTER)
                    flag = True
                except:
                    continue
        except:
            input_box.send_keys(""+Keys.ENTER)
        if flag == False:
            input_box.send_keys("> could not load the horoscope at this time"+Keys.ENTER)
    elif command[1] == "convert":
        try:
            driver.get("https://www.google.co.in/search?q=convert " + user_input.split(' ', 2)[2])
            values = driver.find_elements_by_class_name("vXQmIe")
            tgt_name = driver.find_elements_by_class_name("OR9QXc")
            input_box.send_keys("> " + values[0].get_attribute("value") + " " + selectedoption(
                driver.find_element_by_id("ssSucf").get_attribute("innerHTML")) + " = " + values[1].get_attribute(
                "value") + " " + selectedoption(tgt_name[1].get_attribute("innerHTML"))+Keys.ENTER)
        except:
            input_box.send_keys("> could not convert"+Keys.ENTER)
    elif command[1] == "stocks":
        try:
            driver.get("https://www.google.co.in/search?q=stocks: " + user_input.split(' ', 2)[2])
            info = clean(
                driver.find_element_by_id("knowledge-finance-wholepage__entity-summary").get_attribute(("innerHTML")))
            name = info.split('Disclaimer')[0]
            data = info.split('available ')[1]
            input_box.send_keys("> " + name+Keys.SHIFT, Keys.ENTER)
            list = data.split(' ')
            prev = "a"
            data = "> "
            for x in list:
                if '9' >= prev[-1] >= '0' or prev[-1] == '-' or prev[-1] == '%' or prev[-1] == 'T' or prev[-1] == 'M' or \
                        prev[-1] == 'B':
                     input_box.send_keys(Keys.SHIFT, Keys.ENTER)
                     input_box.send_keys(x)
                else:
                    input_box.send_keys(" " + x)
                prev = x

            input_box.send_keys(Keys.ENTER)
        except:
            input_box.send_keys("> could not find the stocklisting" + Keys.ENTER)
    else:
        input_box.send_keys("> command unknown"+Keys.ENTER)


def chatbot(net, sess, chars, vocab, max_length, beam_width, relevance, temperature, topn):
    states = initial_state_with_relevance_masking(net, sess, relevance)
    driver = webdriver.Chrome('E:\chatbot-rnn-20190504T064221Z-001\cd.exe')
    driver.get("https://web.whatsapp.com/")
    wait = WebDriverWait(driver, 600)
    target = '"Buff"'
    string = "> hello, you are currently talking to an automated artificial converstaion entity. the rules to talk to me are very simple.\n1.use English.proper format\grammer\punctuation is appreciated\n2.use of emojiticons might dicrease the accuracy of meanings so avoid them\n3.a message that starts with '> ' will not be replied to\n4.only last message will be replied to when multiple messages are sent so avoid sending more than one message\n> ###the language used here might be inappropriate, indecent or hurtful. users discretion is adviced###"
    x_arg = '//span[contains(@title,' + target + ')]'
    group_title = wait.until(EC.presence_of_element_located((By.XPATH, x_arg)))
    group_title.click()
    inp_xpath = "//div[@contenteditable='true']"
    input_box = wait.until(EC.presence_of_element_located((
        By.XPATH, inp_xpath)))
    '''
    for i in range(100):
        input_box.send_keys(string + Keys.ENTER)
        Keys.PAGE_UP
        time.sleep(1)
    '''
    input_box.send_keys(string + Keys.ENTER)
    chrome_op = options.Options()
    chrome_op.add_argument("--headless")
    assistdriver = webdriver.Chrome('E:\chatbot-rnn-20190504T064221Z-001\cd.exe', chrome_options=chrome_op)
    # assistdriver = webdriver.Chrome('E:\chatbot-rnn-20190504T064221Z-001\cd.exe')
    msgcounter=0
    while True:
        time.sleep(1)
        msgs = driver.find_elements_by_class_name("selectable-text")
        user_input = msgs[-2].get_attribute('innerHTML')
        if user_input.split(' ', 1)[0] == '&gt;':
            continue
        print(user_input)
        if blacklist(user_input):
            input_box.send_keys("> you are kindly advised to refrain from usage of abusive content" + Keys.ENTER)
            continue
        if user_input[0] == '#':
            assist(user_input,assistdriver,driver,input_box)
            continue
        tts = gTTS(text=user_input, lang='en')
        path='E:\chatbot-rnn-20190504T064221Z-001\\'+str(msgcounter)+".mp3"
        tts.save(path)
        playsound(path)
        os.remove(path)
        msgcounter+=1
        path = "E:\chatbot-rnn-20190504T064221Z-001\\" + str(msgcounter) + ".mp3"
        # input_box.send_keys('Chatbot: msg recieved: ' + msg + Keys.ENTER)
        user_command_entered, reset, states, relevance, temperature, topn, beam_width = process_user_command(
            user_input, states, relevance, temperature, topn, beam_width)
        if reset: states = initial_state_with_relevance_masking(net, sess, relevance)
        if not user_command_entered:
            states = forward_text(net, sess, states, relevance, vocab, sanitize_text(vocab, "> " + user_input + "\n>"))
            computer_response_generator = beam_search_generator(sess=sess, net=net,
                                                                initial_state=copy.deepcopy(states), initial_sample=vocab[' '],
                                                                early_term_token=vocab['\n'], beam_width=beam_width, forward_model_fn=forward_with_mask,
                                                                forward_args={'relevance':relevance, 'mask_reset_token':vocab['\n'], 'forbidden_token':vocab['>'],
                                                                              'temperature':temperature, 'topn':topn})
            out_chars = []
            reply=""
            for i, char_token in enumerate(computer_response_generator):
                out_chars.append(chars[char_token])
                #print(possibly_escaped_char(out_chars), end='', flush=True)
                reply+=possibly_escaped_char(out_chars)
                states = forward_text(net, sess, states, relevance, vocab, chars[char_token])
                if i >= max_length: break
            if blacklist(reply):
                input_box.send_keys("> looks like some strong text was generated by the AI. Sorry of inconvinience but i cant reply to this." + Keys.ENTER)
                print(reply)
                continue
            tts2 = gTTS(text=reply, lang='en-us')
            tts2.save(path)
            playsound(path)
            os.remove(path)
            msgcounter+=1
            input_box.send_keys('> ' + reply + Keys.ENTER)
            states = forward_text(net, sess, states, relevance, vocab, sanitize_text(vocab, "\n> "))

def process_user_command(user_input, states, relevance, temperature, topn, beam_width):
    user_command_entered = False
    reset = False
    try:
        if user_input.startswith('--temperature '):
            user_command_entered = True
            temperature = max(0.001, float(user_input[len('--temperature '):]))
            print("[Temperature set to {}]".format(temperature))
        elif user_input.startswith('--relevance '):
            user_command_entered = True
            new_relevance = float(user_input[len('--relevance '):])
            if relevance <= 0. and new_relevance > 0.:
                states = [states, copy.deepcopy(states)]
            elif relevance > 0. and new_relevance <= 0.:
                states = states[0]
            relevance = new_relevance
            print("[Relevance disabled]" if relevance <= 0. else "[Relevance set to {}]".format(relevance))
        elif user_input.startswith('--topn '):
            user_command_entered = True
            topn = int(user_input[len('--topn '):])
            print("[Top-n filtering disabled]" if topn <= 0 else "[Top-n filtering set to {}]".format(topn))
        elif user_input.startswith('--beam_width '):
            user_command_entered = True
            beam_width = max(1, int(user_input[len('--beam_width '):]))
            print("[Beam width set to {}]".format(beam_width))
        elif user_input.startswith('--reset'):
            user_command_entered = True
            reset = True
            print("[Model state reset]")
    except ValueError:
        print("[Value error with provided argument.]")
    return user_command_entered, reset, states, relevance, temperature, topn, beam_width

def consensus_length(beam_outputs, early_term_token):
    for l in range(len(beam_outputs[0])):
        if l > 0 and beam_outputs[0][l-1] == early_term_token:
            return l-1, True
        for b in beam_outputs[1:]:
            if beam_outputs[0][l] != b[l]: return l, False
    return l, False

def scale_prediction(prediction, temperature):
    if (temperature == 1.0): return prediction # Temperature 1.0 makes no change
    np.seterr(divide='ignore')
    scaled_prediction = np.log(prediction) / temperature
    scaled_prediction = scaled_prediction - np.logaddexp.reduce(scaled_prediction)
    scaled_prediction = np.exp(scaled_prediction)
    np.seterr(divide='warn')
    return scaled_prediction

def forward_with_mask(sess, net, states, input_sample, forward_args):
    # forward_args is a dictionary containing arguments for generating probabilities.
    relevance = forward_args['relevance']
    mask_reset_token = forward_args['mask_reset_token']
    forbidden_token = forward_args['forbidden_token']
    temperature = forward_args['temperature']
    topn = forward_args['topn']

    if relevance <= 0.:
        # No relevance masking.
        prob, states = net.forward_model(sess, states, input_sample)
    else:
        # states should be a 2-length list: [primary net state, mask net state].
        if input_sample == mask_reset_token:
            # Reset the mask probs when reaching mask_reset_token (newline).
            states[1] = initial_state(net, sess)
        primary_prob, states[0] = net.forward_model(sess, states[0], input_sample)
        primary_prob /= sum(primary_prob)
        mask_prob, states[1] = net.forward_model(sess, states[1], input_sample)
        mask_prob /= sum(mask_prob)
        prob = np.exp(np.log(primary_prob) - relevance * np.log(mask_prob))
    # Mask out the forbidden token (">") to prevent the bot from deciding the chat is over)
    prob[forbidden_token] = 0
    # Normalize probabilities so they sum to 1.
    prob = prob / sum(prob)
    # Apply temperature.
    prob = scale_prediction(prob, temperature)
    # Apply top-n filtering if enabled
    if topn > 0:
        prob[np.argsort(prob)[:-topn]] = 0
        prob = prob / sum(prob)
    return prob, states

def beam_search_generator(sess, net, initial_state, initial_sample,
    early_term_token, beam_width, forward_model_fn, forward_args):
    '''Run beam search! Yield consensus tokens sequentially, as a generator;
    return when reaching early_term_token (newline).

    Args:
        sess: tensorflow session reference
        net: tensorflow net graph (must be compatible with the forward_net function)
        initial_state: initial hidden state of the net
        initial_sample: single token (excluding any seed/priming material)
            to start the generation
        early_term_token: stop when the beam reaches consensus on this token
            (but do not return this token).
        beam_width: how many beams to track
        forward_model_fn: function to forward the model, must be of the form:
            probability_output, beam_state =
                    forward_model_fn(sess, net, beam_state, beam_sample, forward_args)
            (Note: probability_output has to be a valid probability distribution!)
        tot_steps: how many tokens to generate before stopping,
            unless already stopped via early_term_token.
    Returns: a generator to yield a sequence of beam-sampled tokens.'''
    # Store state, outputs and probabilities for up to args.beam_width beams.
    # Initialize with just the one starting entry; it will branch to fill the beam
    # in the first step.
    beam_states = [initial_state] # Stores the best activation states
    beam_outputs = [[initial_sample]] # Stores the best generated output sequences so far.
    beam_probs = [1.] # Stores the cumulative normalized probabilities of the beams so far.

    while True:
        # Keep a running list of the best beam branches for next step.
        # Don't actually copy any big data structures yet, just keep references
        # to existing beam state entries, and then clone them as necessary
        # at the end of the generation step.
        new_beam_indices = []
        new_beam_probs = []
        new_beam_samples = []

        # Iterate through the beam entries.
        for beam_index, beam_state in enumerate(beam_states):
            beam_prob = beam_probs[beam_index]
            beam_sample = beam_outputs[beam_index][-1]

            # Forward the model.
            prediction, beam_states[beam_index] = forward_model_fn(
                    sess, net, beam_state, beam_sample, forward_args)

            # Sample best_tokens from the probability distribution.
            # Sample from the scaled probability distribution beam_width choices
            # (but not more than the number of positive probabilities in scaled_prediction).
            count = min(beam_width, sum(1 if p > 0. else 0 for p in prediction))
            best_tokens = np.random.choice(len(prediction), size=count,
                                            replace=False, p=prediction)
            for token in best_tokens:
                prob = prediction[token] * beam_prob
                if len(new_beam_indices) < beam_width:
                    # If we don't have enough new_beam_indices, we automatically qualify.
                    new_beam_indices.append(beam_index)
                    new_beam_probs.append(prob)
                    new_beam_samples.append(token)
                else:
                    # Sample a low-probability beam to possibly replace.
                    np_new_beam_probs = np.array(new_beam_probs)
                    inverse_probs = -np_new_beam_probs + max(np_new_beam_probs) + min(np_new_beam_probs)
                    inverse_probs = inverse_probs / sum(inverse_probs)
                    sampled_beam_index = np.random.choice(beam_width, p=inverse_probs)
                    if new_beam_probs[sampled_beam_index] <= prob:
                        # Replace it.
                        new_beam_indices[sampled_beam_index] = beam_index
                        new_beam_probs[sampled_beam_index] = prob
                        new_beam_samples[sampled_beam_index] = token
        # Replace the old states with the new states, first by referencing and then by copying.
        already_referenced = [False] * beam_width
        new_beam_states = []
        new_beam_outputs = []
        for i, new_index in enumerate(new_beam_indices):
            if already_referenced[new_index]:
                new_beam = copy.deepcopy(beam_states[new_index])
            else:
                new_beam = beam_states[new_index]
                already_referenced[new_index] = True
            new_beam_states.append(new_beam)
            new_beam_outputs.append(beam_outputs[new_index] + [new_beam_samples[i]])
        # Normalize the beam probabilities so they don't drop to zero
        beam_probs = new_beam_probs / sum(new_beam_probs)
        beam_states = new_beam_states
        beam_outputs = new_beam_outputs
        # Prune the agreed portions of the outputs
        # and yield the tokens on which the beam has reached consensus.
        l, early_term = consensus_length(beam_outputs, early_term_token)
        if l > 0:
            for token in beam_outputs[0][:l]: yield token
            beam_outputs = [output[l:] for output in beam_outputs]
        if early_term: return

if __name__ == '__main__':
    main()