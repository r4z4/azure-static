```python
import numpy as np
import regex as re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import utils

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
```


```python
import os
dirpath = '../data/clean_pkl/to_use/'
directory = os.fsencode(dirpath)
df = pd.DataFrame(columns=['newsgroup', 'body'])
for file in os.listdir(directory):
     filename = os.fsdecode(file)
     name, ext = filename.split('.')
     if filename.endswith(".pkl"): 
         df_ = pd.read_pickle(dirpath + filename) 
         df = pd.merge(
            df, df_, how="outer"
        )
         continue
     else:
         continue
```


```python
df.shape
```




    (40590, 2)




```python
df = pd.read_pickle('../data/dataframes/newsgroup_body_cleaned.pkl')
```


```python
from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
print(df.sample(frac=1).reset_index(drop=True).loc[:,['newsgroup', 'body']].head().to_markdown())
```


<style>.container { width:100% !important; }</style>


    |    | newsgroup   | body                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
    |---:|:------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
    |  0 | religion    | deuterocanon esp sirach poram ihlpbattcom wrote let talk principl accept god set standard ought includ scriptur - ask  authorit authorit qualiti write itself- statement commun faith whether accept write norm  prophet propheci defin speak forth god messag much apocrypha must sure qualifi  authent standard authent function author histor accuraci  dynam thi suppos mean mani apocryph book highli dynam -thought provok faith even excit  receiv collect read use cours apocryph book receiv collect els would read use still cathol orthodox church count apocrapha fall short glori god thi demonstr fals quot unger bibl dictionari apocrapha  abound histor geograph inaccuraci anachron book bibl  teach doctrin fals foster practic varianc sacr scriptur fals whose interpret church accept find contradict rest scriptur  resort literari type display artifici subject matter style keep sacr scriptur thi pure subject evalu apocryph book demonstr categori form write found scriptur fact one could argu apocryph addit book esther act rather bring unscripturelik book esther line book  lack distinct element give genuin scriptur divin charact prophet power poetic religi feel ever read wisdom ben sira wisdom solomon exhibit everi bit much poetic religi feel psalm proverb delet view word warn everyon hear word propheci thi book anyon add anyth god add plagu describ thi book anyon take away thi book propheci god take away hi share tree life holi citi rev - sure thi set standard man-mad tradit word clearli meant refer book revel alon whole bodi scriptur revel wa accept veri late canon church simpli see primari role ani kind identifi limit scriptur also noteworthi consid jesu attitud argument pharise ani ot canon john - explain hi follow road emmau law prophet psalm refer - ot divis scriptur luke  well luke  take genesi chronicl jewish order - would say genesi malachi scriptur jesu doe refer canon simpl reason hi day canon establish close collect book apocrypha part septuagint wa bibl earli church hebrew canon wa close  ce torah pentateuch law wa establish jesu day prophet exclusion daniel write howev still flux jesu doe refer write onli psalm part book apocrypha part literatur wa eventu sift separ argu jesu refer jewish canon order luke  weak best quot scriptur tell chronolog stori mention abov hebrew canon especi present order exist jesu day revdak netcomcom |
    |  1 | sport       | oiler rumour - team move press confer next week heard stori local sport news broadcast edmonton oiler owner peter pocklington hold press confer next week exact detail known believ concern oiler futur rumour ha pocklington sign tent leas arrang copp collesium hamilton dure press confer pocklington may announc deal quit possibl deal may simpli way forc edmonton northland renegoti oiler leas stadium northland ha offer buy oiler  million earlier offer wa reject immedi pocklington opinion divid edmonton ha fairli support oiler even though small market team mani sellout  even problem team thi year still brought fan mani team larger citi hand team doe move place deserv hamilton cours would affect grand realign scheme bettman                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
    |  2 | politics    | ban firearm articl apr gnvifasufledu jrm gnvifasufledu write alcohol ban today would much difficult manag large-scal smuggl oper cop rank narrow notch militari commun intellig gather firepow proof assert love uh pleas explain whi smuggler also rank notch abov militari term commun intellig gather eg whi fight offici bribe give hundr grand let semi past firepow similar vein amount marijuana smuggl thi countri ha greatli decreas thi becaus value-per-pound veri low compar cocain heroin simpli worth risk uneconom reefer domest less pressur domest produc showi raid notwithstand thu econom note though domest reefer veri strong small volum goe long way make alcohol stronger  proof - good dollarpound deal point argu black market work doe cours firearm tend fall thi low dollarpound area thi wrong way quantifi thing smuggler would concern valuecub foot go gun show price crate good qualiti handgun would econom smuggl product would local mani peopl local skill motiv assembl worthwhil firearm scratch high-rank crime figur could worthwhil firearm hell anyth work go get copi armi  improvis munit manual see easi make function firearm obtain import uzi averag person averag thug would lucki get zip-gun - would pay nose pay  inconspicu part local k-mart nose drew -- betz gozeridbsuedu brought termin free state idaho outlaw right onli outlaw right spook fodder fema nsa clinton gore insurrect nsc semtex neptunium terrorist cia mi mi kgb deuterium                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
    |  3 | seller      | scan radio realist pro--wa  sell  articl  altradioscann path usenetinscwruedu clevelandfreenetedu aj newsgroup altradioscann realist pro- sale-wa  sell  obo date  apr   gmt organ case western reserv univers cleveland ohio usa line  message-id roo du usenetinscwruedu nntp-posting-host slcinscwruedu hello realist pro- scanner saleher small desc ription  program chanel fulli detail backlight digit display headphon jack antenna jack remov telescop antenna auto search coverag -mhz -mhz -mhz origin cost  sell  thank -- buchanan  fear gover fear gun without nd amend guarante ou r freedom aj clevelandfreenetedu -- buchanan  fear gover fear gun without nd amend guarante ou r freedom aj clevelandfreenetedu                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
    |  4 | comp_elec   | help object appear thrice hey got equat editor sinc nt automag appear object dialog box ie insert -- object -- equat decid manual place went winini anoth way thi embed section ad equationequ equat pathfilenam pictur nt work quit window go back aha mistak correct look fine start window doe nt work play one point two entri see one work thother nt final get work onli thing see differ first item list use last end three equat entri work onli one entri winini doe ani netian know wrong rather correct thi ie make equat appear onc also entri embed appear abov obviou pathfilenam execut whatev pictur ha someth withth way appear picturedescript ie soundrecsound sound whate differ st sound nd soundrec nt think name execut entri eg msworkschart nt thank ia mickey -- pe- michael panayiotaki louray seasgwuedu ace uunet seasgwuedu louray make ms-window grp file reflect hd directori well ai nt alway right never wrong gd                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |



```python
df.shape
```




    (35790, 2)




```python
from itertools import islice
def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

# This will output lists of words, at most 100 each
def long_split(s):
    if len(s) > 100:
        tuple_list = list(chunk(s.split(), 100))
        return [' '.join(tups) for tups in tuple_list]
    
```


```python
df['exploded_body'] = df['body'].apply(lambda x: long_split(x))

```


```python
df['exploded_body'][3]
```




    ['analges diuret sometim see otc prepar muscl achesback ach combin aspirin diuret idea seem reduc inflamm get rid fluid doe thi actual work thank -larri c']




```python
df = df.explode('exploded_body')
```


```python
df.shape
```




    (72250, 3)



## Recheck our corpus

I am still trying get a better sense of what is happening here in the explode function to cause is to get None in the exploded_body col. In cases like this when I just always like to make sure I look over the data to make sure it is still correct. We'll eventually find out when we get our eval metrics but it is pretty clean if you just sample a few rows and make sure you don't have Wayne Gretzky articles under religion.  He was good though.


```python
from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
print(df[df['exploded_body'].apply(lambda x: not isinstance(x, str))].to_markdown())
```


<style>.container { width:100% !important; }</style>


    |       | newsgroup   | body                                                                                                 | exploded_body   |
    |------:|:------------|:-----------------------------------------------------------------------------------------------------|:----------------|
    |    41 | sci_med     | twitch eye thi one time attribut lack sleep sinc disappear night good zzz                            |                 |
    |    56 | sci_med     | open letter hillari rodham clinton  post one repli letter -km                                        |                 |
    |   127 | sci_med     | erythromycin erythromycin effect treat pneumonia -fm                                                 |                 |
    |   285 | sci_med     | foreskin troubl done short circumcis adult male whose foreskin retract                               |                 |
    |   327 | sci_med     | food-rel seizur articl apr spdcccom dyer spdcccom steve dyer write newsgroup                         |                 |
    |   389 | sci_med     | cost roxon doe anyon know approxim prescript cost  ml bottl roxon morphin thank                      |                 |
    |   409 | sci_med     | accupunctur aid friend mine see acupuncturist want know ani danger get aid needl thank -alic         |                 |
    |   446 | sci_med     | angri thi doctor report local bbb better busi bureau bill claussen                                   |                 |
    |   541 | sci_med     | arythmia alexi perri ask low blood potassium could danger ye zz                                      |                 |
    |   561 | sci_med     | choler great ntnf semant war cross-post altpsychologyperson sinc talk physician person apolog        |                 |
    |   568 | sci_med     | food-rel seizur articl cxlr athenacsugaedu mcovingt aisunaiugaedu michael covington write newsgroup  |                 |
    |   680 | sci_med     | help reflux esophag pleas post result close friend ha thi condit ha ask question                     |                 |
    |   722 | sci_med     | hip replac                                                                                           |                 |
    |   744 | sci_med     | discuss altpsychoact seriou discuss drug vs get good bong man whi group moder would elimin idiot     |                 |
    |   757 | sci_med     | sell ten unit sci med peopl sell ten unit doe sold physician liscen person doug opdb vmcclatechedu   |                 |
    |   763 | sci_med     | suppli transplant harvest order                                                                      |                 |
    |   765 | sci_med     | med school admiss alway osteopathi colleg                                                            |                 |
    |   783 | sci_med     | krillean photographi veri good                                                                       |                 |
    |   815 | sci_med     | allerg reaction laser printer                                                                        |                 |
    |   857 | sci_med     | natur altern estrogen need diet diverticular diseas idea gastrointestin distress                     |                 |
    |   858 | sci_med     | sciatica idea relief sciatica pleas respond e-mail                                                   |                 |
    |   913 | sci_med     | lithium question doctor want  year old hi                                                            |                 |
    |   948 | sci_med     | q repel wasp thi cross post recgarden                                                                |                 |
    |   972 | sci_med     | fever blister caus cure fever blister respect request thank -d ian origin anoth permannet kit        |                 |
    |   984 | sci_med     | qualiti manag                                                                                        |                 |
    |   986 | sci_med     | need info circumcis medic con pro bullshit                                                           |                 |
    |  1015 | sci_med     |                                                                                                      |                 |
    |  1026 | sci_med     |                                                                                                      |                 |
    |  1031 | sci_med     | twitch eye thi one time attribut lack sleep sinc disappear night good zzz                            |                 |
    |  1046 | sci_med     | open letter hillari rodham clinton  post one repli letter -km                                        |                 |
    |  1117 | sci_med     | erythromycin erythromycin effect treat pneumonia -fm                                                 |                 |
    |  1275 | sci_med     | foreskin troubl done short circumcis adult male whose foreskin retract                               |                 |
    |  1317 | sci_med     | food-rel seizur articl apr spdcccom dyer spdcccom steve dyer write newsgroup                         |                 |
    |  1379 | sci_med     | cost roxon doe anyon know approxim prescript cost  ml bottl roxon morphin thank                      |                 |
    |  1399 | sci_med     | accupunctur aid friend mine see acupuncturist want know ani danger get aid needl thank -alic         |                 |
    |  1436 | sci_med     | angri thi doctor report local bbb better busi bureau bill claussen                                   |                 |
    |  1531 | sci_med     | arythmia alexi perri ask low blood potassium could danger ye zz                                      |                 |
    |  1551 | sci_med     | choler great ntnf semant war cross-post altpsychologyperson sinc talk physician person apolog        |                 |
    |  1670 | sci_med     | help reflux esophag pleas post result close friend ha thi condit ha ask question                     |                 |
    |  1712 | sci_med     | hip replac                                                                                           |                 |
    |  1734 | sci_med     | discuss altpsychoact seriou discuss drug vs get good bong man whi group moder would elimin idiot     |                 |
    |  1747 | sci_med     | sell ten unit sci med peopl sell ten unit doe sold physician liscen person doug opdb vmcclatechedu   |                 |
    |  1753 | sci_med     | suppli transplant harvest order                                                                      |                 |
    |  1755 | sci_med     | med school admiss alway osteopathi colleg                                                            |                 |
    |  1773 | sci_med     | krillean photographi veri good                                                                       |                 |
    |  1805 | sci_med     | allerg reaction laser printer                                                                        |                 |
    |  1847 | sci_med     | natur altern estrogen need diet diverticular diseas idea gastrointestin distress                     |                 |
    |  1848 | sci_med     | sciatica idea relief sciatica pleas respond e-mail                                                   |                 |
    |  1903 | sci_med     | lithium question doctor want  year old hi                                                            |                 |
    |  1938 | sci_med     | q repel wasp thi cross post recgarden                                                                |                 |
    |  1962 | sci_med     | fever blister caus cure fever blister respect request thank -d ian origin anoth permannet kit        |                 |
    |  1974 | sci_med     | qualiti manag                                                                                        |                 |
    |  1976 | sci_med     | need info circumcis medic con pro bullshit                                                           |                 |
    |  1985 | sci_med     | diff                                                                                                 |                 |
    |  1997 | sci_med     | astrospac frequent seen acronym archive-nam spaceacronym edit  acronym list sciastro                 |                 |
    |  2049 | sci_med     | nuclear wast thank updat                                                                             |                 |
    |  2094 | sci_med     | question titan iv arian  articl corqg newscsouiucedu gwg uxacsouiucedu garret w gengler write        |                 |
    |  2103 | sci_med     | sixty-two thousand wa mani read                                                                      |                 |
    |  2107 | sci_med     | mani read                                                                                            |                 |
    |  2156 | sci_med     | read                                                                                                 |                 |
    |  2410 | sci_med     | planet still imag orbit ether twist must ship good eau clair acid california tom freebairn           |                 |
    |  2464 | sci_med     | new planetkuip object found new kuiper belt object call karla next one call smiley jame nicol        |                 |
    |  2472 | sci_med     | weekli remind frequent ask question list thi notic post weekli                                       |                 |
    |  2492 | sci_med     | vandal sky newsgroup sciastro                                                                        |                 |
    |  2497 | sci_med     | dream degre wa crazi imaginit higgin fnalffnalgov bill higgin -- beam jockey write like involv       |                 |
    |  2513 | sci_med     | planet still imag orbit ether twist pleas get real life                                              |                 |
    |  2627 | sci_med     | vandal sky come thi                                                                                  |                 |
    |  2668 | sci_med     | internet resourc exit                                                                                |                 |
    |  2685 | sci_med     | test pleas ignor                                                                                     |                 |
    |  2729 | sci_med     | interstellar -k note plutocharon talk pluto mission                                                  |                 |
    |  2838 | sci_med     | gamma ray burster follow discuss grb go sciastro discuss detail refer even                           |                 |
    |  2870 | sci_med     | franc spi us mena way french intellieg agent steal document us corpor execut pat                     |                 |
    |  2899 | sci_med     | gp launch next gp launch schedul june th origin va astronomi club --                                 |                 |
    |  2902 | sci_med     | henri spencer anyway cam hawkadiedozau master write etoyoc lelandstanfordedu aaron thode write track |                 |
    |  2917 | sci_med     | dc-x public even better make pete conrad martian suit get ou throw footbal ref                       |                 |
    |  2929 | sci_med     | sdio kaput mention liber presid nixon ford reagan bush noth support true commerci space activ pat    |                 |
    |  2930 | sci_med     | near miss asteroid q trri skywatch project arizona pat                                               |                 |
    |  2944 | sci_med     | space market would wonderful see emblazen across even sky -- thi space rent                          |                 |
    |  2972 | sci_med     | diff                                                                                                 |                 |
    |  2984 | sci_med     | astrospac frequent seen acronym archive-nam spaceacronym edit  acronym list sciastro                 |                 |
    |  3036 | sci_med     | nuclear wast thank updat                                                                             |                 |
    |  3081 | sci_med     | question titan iv arian  articl corqg newscsouiucedu gwg uxacsouiucedu garret w gengler write        |                 |
    |  3090 | sci_med     | sixty-two thousand wa mani read                                                                      |                 |
    |  3092 | sci_med     | mani read                                                                                            |                 |
    |  3101 | sci_med     | mani read                                                                                            |                 |
    |  3108 | sci_med     | mani read                                                                                            |                 |
    |  3113 | sci_med     | mani read                                                                                            |                 |
    |  3123 | sci_med     | mani read                                                                                            |                 |
    |  3125 | sci_med     | sixty-two thousand wa mani read                                                                      |                 |
    |  3138 | sci_med     | read                                                                                                 |                 |
    |  3142 | sci_med     | mani read                                                                                            |                 |
    |  3207 | sci_med     | sixty-two thousand wa mani read                                                                      |                 |
    |  3249 | sci_med     | read                                                                                                 |                 |
    |  3258 | sci_med     | mani read                                                                                            |                 |
    |  3395 | sci_med     | mani read                                                                                            |                 |
    |  3396 | sci_med     | planet still imag orbit ether twist must ship good eau clair acid california tom freebairn           |                 |
    |  3450 | sci_med     | new planetkuip object found new kuiper belt object call karla next one call smiley jame nicol        |                 |
    |  3458 | sci_med     | weekli remind frequent ask question list thi notic post weekli                                       |                 |
    |  3478 | sci_med     | vandal sky newsgroup sciastro                                                                        |                 |
    |  3482 | sci_med     | dream degre wa crazi imaginit higgin fnalffnalgov bill higgin -- beam jockey write like involv       |                 |
    |  3498 | sci_med     | planet still imag orbit ether twist pleas get real life                                              |                 |
    |  3516 | sci_med     | mani read                                                                                            |                 |
    |  3605 | sci_med     | vandal sky newsgroup sciastro                                                                        |                 |
    |  3614 | sci_med     | vandal sky come thi                                                                                  |                 |
    |  3655 | sci_med     | internet resourc exit                                                                                |                 |
    |  3672 | sci_med     | test pleas ignor                                                                                     |                 |
    |  3716 | sci_med     | interstellar -k note plutocharon talk pluto mission                                                  |                 |
    |  3825 | sci_med     | gamma ray burster follow discuss grb go sciastro discuss detail refer even                           |                 |
    |  3857 | sci_med     | franc spi us mena way french intellieg agent steal document us corpor execut pat                     |                 |
    |  3886 | sci_med     | gp launch next gp launch schedul june th origin va astronomi club --                                 |                 |
    |  3904 | sci_med     | dc-x public even better make pete conrad martian suit get ou throw footbal ref                       |                 |
    |  3916 | sci_med     | sdio kaput mention liber presid nixon ford reagan bush noth support true commerci space activ pat    |                 |
    |  3917 | sci_med     | near miss asteroid q trri skywatch project arizona pat                                               |                 |
    |  3931 | sci_med     | space market would wonderful see emblazen across even sky -- thi space rent                          |                 |
    |  3959 | seller      | test                                                                                                 |                 |
    |  3967 | seller      | forsal soni d- diskman newsgroup recaudio                                                            |                 |
    |  3991 | seller      | want futon look larg futon frame call peter - e-mail khiet cnecn                                     |                 |
    |  4081 | seller      | chemic sale chemic gone thank respons omar                                                           |                 |
    |  4121 | seller      | monitor traci monitor way mike damico                                                                |                 |
    |  4132 | seller      | termin forsal                                                                                        |                 |
    |  4133 | seller      | termin sale vt vt compat termin  extern hyess modem amber screen  keyboard cabl make offer           |                 |
    |  4135 | seller      | speaker sale sale bose a subwoof  month old  advent mini  month old email offer craigb rpiedu        |                 |
    |  4152 | seller      | dbase iv sale price drop dbase iv ver   disk manual still shrinkwrap registr materi present ask      |                 |
    |  4162 | seller      | trade k modem pcxt email repli danj holonetnet                                                       |                 |
    |  4172 | seller      | want cheap use gameboytg- game titl say cheap use gameboy tg-  player game pleas email offer rohit   |                 |
    |  4198 | seller      | monitor sale sale kfc svga monitor x dp non-interlac  screen still warranti  best offer              |                 |
    |  4224 | seller      | want refriger want refriger contact  - jamesl galaxynsccom                                           |                 |
    |  4230 | seller      | baud extern modem  mint box manual phonecord  ship denni                                             |                 |
    |  4289 | seller      | unix pc softwar sale                                                                                 |                 |
    |  4304 | seller      | jazz cd  saletrad sell  ca nt realli offer  thi point thank jon                                      |                 |
    |  4305 | seller      | classic cd  sale hey ca nt send mail could pleas resend address lost h moscow thank jon              |                 |
    |  4314 | seller      | eric bosco eric send email address lost reconsid kevin                                               |                 |
    |  4344 | seller      | updat hard drive vga etc articl crosspost                                                            |                 |
    |  4346 | seller      | hp calcul  greet hp  forsal come case manual excel condit ask  interest pleas e-mail today al        |                 |
    |  4418 | seller      | us robot  modem repli haljordan delphicom call  - us robot  dual standard v bi k baud  hst price     |                 |
    |  4432 | seller      | buick centuri estat wagon thi articl wa probabl gener buggi news reader                              |                 |
    |  4456 | seller      | want fax machin sub say thnx tatsuy                                                                  |                 |
    |  4466 | seller      | vh video sale thi movi sold mcdonald  new                                                            |                 |
    |  4477 | seller      | test test                                                                                            |                 |
    |  4495 | seller      | want ide hard drive  vga monitor e-mail                                                              |                 |
    |  4500 | seller      | polk s forsal pair polk s sale brand new never open  craigb rpiedu                                   |                 |
    |  4521 | seller      | roland juno- synthes uniden radar detector  sale actual synth use jump wa oberheim watch video kevin |                 |
    |  4556 | seller      | metallica sale addit sorri forgot add jap import andi                                                |                 |
    |  4560 | seller      | nikkor - af sale  nikkor - af immedi sale excel condit send e-mail detail                            |                 |
    |  4575 | seller      | nikon l camera  nikon l af camera  len camera case packag  send e-mail                               |                 |
    |  4576 | seller      | texa instument ti- calcul texa instrument ti- calculalor excel scientif calcul best offer            |                 |
    |  4606 | seller      | minolta fd  mm forsal minolta fd  mm len sale good condit ask  rupindang dartmouthedu                |                 |
    |  4611 | seller      | mb simm sale  pin n mac includ ship                                                                  |                 |
    |  4622 | seller      | probesplug oscilloscopefunct gener greet sorri typo clipper hook al plug suppos black red mini-      |                 |
    |  4658 | seller      | want  polaroid palett - manual newsgroup miscwant                                                    |                 |
    |  4659 | seller      | nikkor - af forsal nikkor af - f- zoom len excel condit look get  version sell thi ask  offer pleas  |                 |
    |  4665 | seller      | cd sale paid  cd fool got rip                                                                        |                 |
    |  4666 | seller      | sale high-gual conif oil russia ton ton inguiri address er eridanchuvashiasu                         |                 |
    |  4667 | seller      | sale high-gual conif oil russia ton  ton inguiri address er eridanchuvashiasu                        |                 |
    |  4682 | seller      | keyboard want look buy  work keyboard  system prefer  layout look spend  -- david                    |                 |
    |  4693 | seller      | stereo system sale ken                                                                               |                 |
    |  4697 | seller      | nikkor -af forsal  price reduc ask  onli                                                             |                 |
    |  4713 | seller      | forsal wed dress -- size  size  wed dress lot bead inquir -- mst utah oo paid ask o                  |                 |
    |  4715 | seller      | want k n sip want -  k n sip ani pleas tell much want includ ship                                    |                 |
    |  4717 | seller      | look diffract grate glass quantiti yeah sum look place sell diffract grate goggl quantiti thank      |                 |
    |  4718 | seller      | hp plotter sale new plotter  em straight box doc lost make offer cod ship jj uk mikukyedu            |                 |
    |  4722 | seller      | soni camcord  soni ccd-v mm camcord origin bought  onli  origin box accesori jaf andrewcmuedu --     |                 |
    |  4725 | seller      | meg seagat ide hard drive                                                                            |                 |
    |  4740 | seller      | sale sale trident  meg video card x  color  dollar best offer e-mail dcassen mcsdcsumredu            |                 |
    |  4748 | seller      | test thi test thank                                                                                  |                 |
    |  4763 | seller      | want technic  turntabl simpl eh rather get em use new must guarante                                  |                 |
    |  4768 | seller      | nintendo system w super mario  ship obo                                                              |                 |
    |  4786 | seller      | dx mhz motherboard sale                                                                              |                 |
    |  4787 | seller      | lotu -- forsal extra copi lotu -- ver  like get  pleas repli e-mail jth bachudeledu thank jay        |                 |
    |  4788 | seller      | dx mhz motherboard sale hate post dx mhz mb w meg n ram make offer michael                           |                 |
    |  4837 | seller      | cd sale im interset bu student rit pleas repli say contact ed                                        |                 |
    |  4840 | seller      | maxtor mb scsi maxtor xt  mb scsi drive m access time  year old extern case  jaf andrewcmuedu --     |                 |
    |  4849 | seller      | want sim citi pc hello look sim citi pc newus pleas make offer cchu udeledu thank chu                |                 |
    |  4863 | seller      | stereo lp record sale                                                                                |                 |
    |  4876 | seller      | mfm hd mb  mb  seagat st- hh mb  ibm fh mb  mfm type good work condit buyer pay ship ailin --        |                 |
    |  4878 | seller      | xt keyboard  onlyk titl say ibm brand eric                                                           |                 |
    |  4881 | seller      | nice telecop sale tasco eb x-xmm use onc look like new worth  sell  onli buyer pay ship ailin --     |                 |
    |  4901 | seller      | stereo lp sale updat list                                                                            |                 |
    |  4902 | seller      | lp sale                                                                                              |                 |
    |  4917 | seller      | cd rom ibm - cd-rom drive m drive onli make offer trade                                              |                 |
    |  4931 | seller      | test                                                                                                 |                 |
    |  4939 | seller      | forsal soni d- diskman newsgroup recaudio                                                            |                 |
    |  4963 | seller      | want futon look larg futon frame call peter - e-mail khiet cnecn                                     |                 |
    |  5053 | seller      | chemic sale chemic gone thank respons omar                                                           |                 |
    |  5093 | seller      | monitor traci monitor way mike damico                                                                |                 |
    |  5104 | seller      | termin forsal                                                                                        |                 |
    |  5105 | seller      | termin sale vt vt compat termin  extern hyess modem amber screen  keyboard cabl make offer           |                 |
    |  5107 | seller      | speaker sale sale bose a subwoof  month old  advent mini  month old email offer craigb rpiedu        |                 |
    |  5124 | seller      | dbase iv sale price drop dbase iv ver   disk manual still shrinkwrap registr materi present ask      |                 |
    |  5134 | seller      | trade k modem pcxt email repli danj holonetnet                                                       |                 |
    |  5144 | seller      | want cheap use gameboytg- game titl say cheap use gameboy tg-  player game pleas email offer rohit   |                 |
    |  5170 | seller      | monitor sale sale kfc svga monitor x dp non-interlac  screen still warranti  best offer              |                 |
    |  5196 | seller      | want refriger want refriger contact  - jamesl galaxynsccom                                           |                 |
    |  5202 | seller      | baud extern modem  mint box manual phonecord  ship denni                                             |                 |
    |  5261 | seller      | unix pc softwar sale                                                                                 |                 |
    |  5276 | seller      | jazz cd  saletrad sell  ca nt realli offer  thi point thank jon                                      |                 |
    |  5277 | seller      | classic cd  sale hey ca nt send mail could pleas resend address lost h moscow thank jon              |                 |
    |  5286 | seller      | eric bosco eric send email address lost reconsid kevin                                               |                 |
    |  5316 | seller      | updat hard drive vga etc articl crosspost                                                            |                 |
    |  5318 | seller      | hp calcul  greet hp  forsal come case manual excel condit ask  interest pleas e-mail today al        |                 |
    |  5390 | seller      | us robot  modem repli haljordan delphicom call  - us robot  dual standard v bi k baud  hst price     |                 |
    |  5404 | seller      | buick centuri estat wagon thi articl wa probabl gener buggi news reader                              |                 |
    |  5428 | seller      | want fax machin sub say thnx tatsuy                                                                  |                 |
    |  5438 | seller      | vh video sale thi movi sold mcdonald  new                                                            |                 |
    |  5449 | seller      | test test                                                                                            |                 |
    |  5467 | seller      | want ide hard drive  vga monitor e-mail                                                              |                 |
    |  5472 | seller      | polk s forsal pair polk s sale brand new never open  craigb rpiedu                                   |                 |
    |  5528 | seller      | metallica sale addit sorri forgot add jap import andi                                                |                 |
    |  5532 | seller      | nikkor - af sale  nikkor - af immedi sale excel condit send e-mail detail                            |                 |
    |  5547 | seller      | nikon l camera  nikon l af camera  len camera case packag  send e-mail                               |                 |
    |  5548 | seller      | texa instument ti- calcul texa instrument ti- calculalor excel scientif calcul best offer            |                 |
    |  5578 | seller      | minolta fd  mm forsal minolta fd  mm len sale good condit ask  rupindang dartmouthedu                |                 |
    |  5583 | seller      | mb simm sale  pin n mac includ ship                                                                  |                 |
    |  5594 | seller      | probesplug oscilloscopefunct gener greet sorri typo clipper hook al plug suppos black red mini-      |                 |
    |  5630 | seller      | want  polaroid palett - manual newsgroup miscwant                                                    |                 |
    |  5637 | seller      | cd sale paid  cd fool got rip                                                                        |                 |
    |  5638 | seller      | sale high-gual conif oil russia ton ton inguiri address er eridanchuvashiasu                         |                 |
    |  5639 | seller      | sale high-gual conif oil russia ton  ton inguiri address er eridanchuvashiasu                        |                 |
    |  5654 | seller      | keyboard want look buy  work keyboard  system prefer  layout look spend  -- david                    |                 |
    |  5665 | seller      | stereo system sale ken                                                                               |                 |
    |  5669 | seller      | nikkor -af forsal  price reduc ask  onli                                                             |                 |
    |  5685 | seller      | forsal wed dress -- size  size  wed dress lot bead inquir -- mst utah oo paid ask o                  |                 |
    |  5687 | seller      | want k n sip want -  k n sip ani pleas tell much want includ ship                                    |                 |
    |  5689 | seller      | look diffract grate glass quantiti yeah sum look place sell diffract grate goggl quantiti thank      |                 |
    |  5690 | seller      | hp plotter sale new plotter  em straight box doc lost make offer cod ship jj uk mikukyedu            |                 |
    |  5694 | seller      | soni camcord  soni ccd-v mm camcord origin bought  onli  origin box accesori jaf andrewcmuedu --     |                 |
    |  5697 | seller      | meg seagat ide hard drive                                                                            |                 |
    |  5712 | seller      | sale sale trident  meg video card x  color  dollar best offer e-mail dcassen mcsdcsumredu            |                 |
    |  5720 | seller      | test thi test thank                                                                                  |                 |
    |  5735 | seller      | want technic  turntabl simpl eh rather get em use new must guarante                                  |                 |
    |  5740 | seller      | nintendo system w super mario  ship obo                                                              |                 |
    |  5758 | seller      | dx mhz motherboard sale                                                                              |                 |
    |  5759 | seller      | lotu -- forsal extra copi lotu -- ver  like get  pleas repli e-mail jth bachudeledu thank jay        |                 |
    |  5760 | seller      | dx mhz motherboard sale hate post dx mhz mb w meg n ram make offer michael                           |                 |
    |  5809 | seller      | cd sale im interset bu student rit pleas repli say contact ed                                        |                 |
    |  5812 | seller      | maxtor mb scsi maxtor xt  mb scsi drive m access time  year old extern case  jaf andrewcmuedu --     |                 |
    |  5821 | seller      | want sim citi pc hello look sim citi pc newus pleas make offer cchu udeledu thank chu                |                 |
    |  5835 | seller      | stereo lp record sale                                                                                |                 |
    |  5848 | seller      | mfm hd mb  mb  seagat st- hh mb  ibm fh mb  mfm type good work condit buyer pay ship ailin --        |                 |
    |  5850 | seller      | xt keyboard  onlyk titl say ibm brand eric                                                           |                 |
    |  5853 | seller      | nice telecop sale tasco eb x-xmm use onc look like new worth  sell  onli buyer pay ship ailin --     |                 |
    |  5873 | seller      | stereo lp sale updat list                                                                            |                 |
    |  5874 | seller      | lp sale                                                                                              |                 |
    |  5889 | seller      | cd rom ibm - cd-rom drive m drive onli make offer trade                                              |                 |
    |  5905 | politics    | waco militia assembl dumb move smart move would sneak someon tv camera video transmitt john nagl     |                 |
    |  6006 | politics    | arm citizen - april  iftccu                                                                          |                 |
    |  6008 | politics    | proper gun control proper gun control wa gun like american express card iftccu                       |                 |
    |  6018 | politics    | non-leth altern handgun iftccu                                                                       |                 |
    |  6020 | politics    | gun lover wa gun like american express card iftccu                                                   |                 |
    |  6021 | politics    | pill deer hunt iftccu                                                                                |                 |
    |  6066 | politics    | nd amend dead - good excerpt netnew                                                                  |                 |
    |  6084 | politics    | proper gun control proper gun control wa gun like american express card iftccu                       |                 |
    |  6146 | politics    | chang name  make new newsgroup call                                                                  |                 |
    |  6172 | politics    | atf burn dividian ranch survivor nut case panic jump gun net befor get fact straight                 |                 |
    |  6208 | politics    | cnn sale anyon keep list potenti contributor put  condit abov keith emmen kde boihpcom               |                 |
    |  6273 | politics    | handgun restrict bbsbilland tsoftnet handgun restrict newsgroup                                      |                 |
    |  6312 | politics    | knew would happen                                                                                    |                 |
    |  6354 | politics    | articl apr iitmaxiitedu draughn iitmaxiitedu mark draughn write followup                             |                 |
    |  6376 | politics    | slaughter followup                                                                                   |                 |
    |  6383 | politics    | batf fbi right thing waco ditto great post joekusmierczak mailtrincolledu                            |                 |
    |  6411 | politics    | chang name suggest anoth name chang thoma parsli vidkun quisl                                        |                 |
    |  6458 | politics    | gun gone good riddanc iftccu                                                                         |                 |
    |  6470 | politics    | waco cross-post                                                                                      |                 |
    |  6563 | politics    | evil tax dollar work wa atf burn ranch etc etc lord hope nt hoover wa pro wa monstrou dan            |                 |
    |  6609 | politics    | waco aflam abolish cult start fbi                                                                    |                 |
    |  6613 | politics    | fed caught anoth lie -- pete norton peten wellsfcau peten holonetnet norton houamococom              |                 |
    |  6625 | politics    | cnn sale count  allan lockridg opinion sale -- allan lockridg -- allanl                              |                 |
    |  6708 | politics    | dayton gun buy back boston gun buy back excerpt netnew                                               |                 |
    |  6712 | politics    | batf acronym b urn f ucker                                                                           |                 |
    |  6762 | politics    | chang name read read post quisl look dictionari nt read thi thoma                                    |                 |
    |  6815 | politics    | waco militia assembl dumb move smart move would sneak someon tv camera video transmitt john nagl     |                 |
    |  6916 | politics    | arm citizen - april  iftccu                                                                          |                 |
    |  6918 | politics    | proper gun control proper gun control wa gun like american express card iftccu                       |                 |
    |  6919 | politics    | proper gun control proper gun control wa gun like american express card iftccu                       |                 |
    |  6928 | politics    | non-leth altern handgun iftccu                                                                       |                 |
    |  6930 | politics    | gun lover wa gun like american express card iftccu                                                   |                 |
    |  6931 | politics    | pill deer hunt iftccu                                                                                |                 |
    |  6976 | politics    | nd amend dead - good excerpt netnew                                                                  |                 |
    |  6994 | politics    | proper gun control proper gun control wa gun like american express card iftccu                       |                 |
    |  7056 | politics    | chang name  make new newsgroup call                                                                  |                 |
    |  7082 | politics    | atf burn dividian ranch survivor nut case panic jump gun net befor get fact straight                 |                 |
    |  7118 | politics    | cnn sale anyon keep list potenti contributor put  condit abov keith emmen kde boihpcom               |                 |
    |  7183 | politics    | handgun restrict bbsbilland tsoftnet handgun restrict newsgroup                                      |                 |
    |  7222 | politics    | knew would happen                                                                                    |                 |
    |  7264 | politics    | articl apr iitmaxiitedu draughn iitmaxiitedu mark draughn write followup                             |                 |
    |  7286 | politics    | slaughter followup                                                                                   |                 |
    |  7293 | politics    | batf fbi right thing waco ditto great post joekusmierczak mailtrincolledu                            |                 |
    |  7321 | politics    | chang name suggest anoth name chang thoma parsli vidkun quisl                                        |                 |
    |  7368 | politics    | gun gone good riddanc iftccu                                                                         |                 |
    |  7380 | politics    | waco cross-post                                                                                      |                 |
    |  7473 | politics    | evil tax dollar work wa atf burn ranch etc etc lord hope nt hoover wa pro wa monstrou dan            |                 |
    |  7519 | politics    | waco aflam abolish cult start fbi                                                                    |                 |
    |  7523 | politics    | fed caught anoth lie -- pete norton peten wellsfcau peten holonetnet norton houamococom              |                 |
    |  7535 | politics    | cnn sale count  allan lockridg opinion sale -- allan lockridg -- allanl                              |                 |
    |  7618 | politics    | dayton gun buy back boston gun buy back excerpt netnew                                               |                 |
    |  7622 | politics    | batf acronym b urn f ucker                                                                           |                 |
    |  7672 | politics    | chang name read read post quisl look dictionari nt read thi thoma                                    |                 |
    |  7757 | politics    | wa go hezbollah realli tri understand brad becaus appear kill know jess                              |                 |
    |  7939 | politics    | rejoind question isra bc clevelandfreenetedu mark ira kaufman newsgroup                              |                 |
    |  8158 | politics    | ajerk good case right abort system fourdcom phone -- cute quot comput mean never say sorri           |                 |
    |  8159 | politics    | argic aswer one question get retard system fourdcom phone -- cute quot comput mean never say sorri   |                 |
    |  8180 | politics    | happi birthday israel israel - happi th birthday                                                     |                 |
    |  8199 | politics    | isra terror kid                                                                                      |                 |
    |  8215 | politics    | uva wow sad see univers virginia ha begun produc virul breed jew-hat self-hat jew roar lion roar     |                 |
    |  8226 | politics    | uva think kind uncal blanket statement caus censorship mr jefferson univers wrong                    |                 |
    |  8249 | politics    | zionism racism ye want read articl                                                                   |                 |
    |  8260 | politics    | wayn mcguir someon prove anon anonpenetfi ran restock pcp miss sniff                                 |                 |
    |  8263 | politics    | hamza salah humanist hey want post forward ca nt get sysadmin pay ani attent                         |                 |
    |  8336 | politics    | serdar loser system fourdcom phone -- cute quot comput mean never say sorri                          |                 |
    |  8337 | politics    | serdar hey serdar retard system fourdcom phone -- cute quot comput mean never say sorri              |                 |
    |  8338 | politics    | serdar anal retent wimp system fourdcom phone -- cute quot comput mean never say sorri               |                 |
    |  8350 | politics    | argic one word loser system fourdcom phone -- cute quot comput mean never say sorri                  |                 |
    |  8383 | politics    | civil shut andi                                                                                      |                 |
    |  8429 | politics    | illeg post szljubi ucdavi reader                                                                     |                 |
    |  8475 | politics    | friend tpm send greet thi outrag nt even dog                                                         |                 |
    |  8530 | politics    | error condit want subscrib locat israel name david gotlieb                                           |                 |
    |  8575 | politics    | west bank basebal ha report nation basebal leagu ha spot west bank recruit pitcher --                |                 |
    |  8635 | politics    | saudi clergi condemn debut human right group excerpt netnew                                          |                 |
    |  8656 | politics    | netanyahu stone ana omran sinc jew way find leader follow ana much netanyahu pay write thi           |                 |
    |  8697 | politics    | wa go hezbollah realli tri understand brad becaus appear kill know jess                              |                 |
    |  8879 | politics    | rejoind question isra bc clevelandfreenetedu mark ira kaufman newsgroup                              |                 |
    |  9098 | politics    | ajerk good case right abort system fourdcom phone -- cute quot comput mean never say sorri           |                 |
    |  9099 | politics    | argic aswer one question get retard system fourdcom phone -- cute quot comput mean never say sorri   |                 |
    |  9120 | politics    | happi birthday israel israel - happi th birthday                                                     |                 |
    |  9139 | politics    | isra terror kid                                                                                      |                 |
    |  9155 | politics    | uva wow sad see univers virginia ha begun produc virul breed jew-hat self-hat jew roar lion roar     |                 |
    |  9166 | politics    | uva think kind uncal blanket statement caus censorship mr jefferson univers wrong                    |                 |
    |  9189 | politics    | zionism racism ye want read articl                                                                   |                 |
    |  9200 | politics    | wayn mcguir someon prove anon anonpenetfi ran restock pcp miss sniff                                 |                 |
    |  9203 | politics    | hamza salah humanist hey want post forward ca nt get sysadmin pay ani attent                         |                 |
    |  9276 | politics    | serdar loser system fourdcom phone -- cute quot comput mean never say sorri                          |                 |
    |  9277 | politics    | serdar hey serdar retard system fourdcom phone -- cute quot comput mean never say sorri              |                 |
    |  9278 | politics    | serdar anal retent wimp system fourdcom phone -- cute quot comput mean never say sorri               |                 |
    |  9290 | politics    | argic one word loser system fourdcom phone -- cute quot comput mean never say sorri                  |                 |
    |  9323 | politics    | civil shut andi                                                                                      |                 |
    |  9369 | politics    | illeg post szljubi ucdavi reader                                                                     |                 |
    |  9415 | politics    | friend tpm send greet thi outrag nt even dog                                                         |                 |
    |  9470 | politics    | error condit want subscrib locat israel name david gotlieb                                           |                 |
    |  9515 | politics    | west bank basebal ha report nation basebal leagu ha spot west bank recruit pitcher --                |                 |
    |  9575 | politics    | saudi clergi condemn debut human right group excerpt netnew                                          |                 |
    |  9596 | politics    | netanyahu stone ana omran sinc jew way find leader follow ana much netanyahu pay write thi           |                 |
    |  9615 | politics    | top ten reason aid russian wa nt tricki dick issu stern warn bush clinton lose russia la lost china  |                 |
    |  9635 | politics    | whi doe clayton cramer fixat molest children articl apr armorycom                                    |                 |
    |  9656 | politics    | e-mail doe anyon e-mail address white hous pleas send thank lot                                      |                 |
    |  9703 | politics    | celebr liberti  narr narr narr cb                                                                    |                 |
    |  9706 | politics    | email doe anyon prez clinton e-mail address thank lot                                                |                 |
    |  9749 | politics    | thought commerci advertis wa allow debat delet guess allow                                           |                 |
    |  9800 | politics    | karadz bosnia peac plan doe anyon think judg wopner would karadz wa trial befor nevah happen thought |                 |
    |  9803 | politics    | model unit nation observ nation model unit nation nyc one word awsom peac matt                       |                 |
    |  9991 | politics    | welcom polic state usa articl apr ccusuedu slpk ccusuedu write xref dscomsa altactivism              |                 |
    | 10047 | politics    | race violenc thi                                                                                     |                 |
    | 10051 | politics    | sexual proposit sexual harass thi tesrt                                                              |                 |
    | 10134 | politics    | carpet bosnia anybodi carpet bosniaserbia mean like carpet bomb serbian posit                        |                 |
    | 10148 | politics    | itali next domino fall note cross-post altpoliticsitali                                              |                 |
    | 10288 | politics    | major burger chain offer                                                                             |                 |
    | 10410 | politics    | whi doe clayton cramer fixat molest children articl apr armorycom                                    |                 |
    | 10431 | politics    | e-mail doe anyon e-mail address white hous pleas send thank lot                                      |                 |
    | 10478 | politics    | celebr liberti  narr narr narr cb                                                                    |                 |
    | 10481 | politics    | email doe anyon prez clinton e-mail address thank lot                                                |                 |
    | 10524 | politics    | thought commerci advertis wa allow debat delet guess allow                                           |                 |
    | 10578 | politics    | model unit nation observ nation model unit nation nyc one word awsom peac matt                       |                 |
    | 10766 | politics    | welcom polic state usa articl apr ccusuedu slpk ccusuedu write xref dscomsa altactivism              |                 |
    | 10822 | politics    | race violenc thi                                                                                     |                 |
    | 10826 | politics    | sexual proposit sexual harass thi tesrt                                                              |                 |
    | 10909 | politics    | carpet bosnia anybodi carpet bosniaserbia mean like carpet bomb serbian posit                        |                 |
    | 10923 | politics    | itali next domino fall note cross-post altpoliticsitali                                              |                 |
    | 11063 | politics    | major burger chain offer                                                                             |                 |
    | 11153 | sport       | giant win pennant giant win pennant giant win pennant gi ooop guess littl earli see octob            |                 |
    | 11160 | sport       | dodger newslett could somebodi pleas tell dodger newslett net subscrib thank joel                    |                 |
    | 11191 | sport       | red sox win st bosox  royal  wp clemen - lp appier - key hit mike greenwel  trippl base load         |                 |
    | 11197 | sport       | vega odd doe anyon list vega odd team make world seri appreci mail thank rickc corpsgicom            |                 |
    | 11227 | sport       | david well ha david well land team yet think tiger anem pitch would grab thi guy pronto dc           |                 |
    | 11280 | sport       | sax ani news steve statu sinc lost start job would appreci thank gwyn                                |                 |
    | 11344 | sport       | cub expo roster question make room harkey cub sent shawn boski aaa                                   |                 |
    | 11348 | sport       | texa ranger ticket info would someon pleas give address texa ranger ticket order thank veri much jim |                 |
    | 11405 | sport       | kevin roger hpcc                                                                                     |                 |
    | 11412 | sport       | fenway gif love see shea stadium gif -sean behind bag - vin sculli                                   |                 |
    | 11422 | sport       | nl vs al doug robert - ken hill nl mvp let go spo                                                    |                 |
    | 11428 | sport       | scott erickson doe anyon scoop scot erickson long go                                                 |                 |
    | 11431 | sport       | jewish basebal player dave kingman jewish bob comarow eisnerdecusorg                                 |                 |
    | 11449 | sport       | request al stat anyon al individu stat find k --                                                     |                 |
    | 11464 | sport       | twin game doe anyon know twin game broadcast good ole ame iowa thank                                 |                 |
    | 11512 | sport       | doe fred mcgriff padr becom free agent                                                               |                 |
    | 11538 | sport       | ryam - week nolan ryan ha torn cartlidg inhi right knee surgeri expect miss - week --                |                 |
    | 11541 | sport       | yanke fear rawley eastwick                                                                           |                 |
    | 11572 | sport       | jewish basebal player add steve rosenberg one-tim white sox reliev met system list greg              |                 |
    | 11589 | sport       | think go cri yanke lose ca nt believ thi howe ha era  improv key pitch great game screw              |                 |
    | 11595 | sport       | astro real think astro go place current first place - - road                                         |                 |
    | 11601 | sport       | oriol trivia bunker mcnalli later pappa estrada steve barber jay                                     |                 |
    | 11606 | sport       | jay game anyon know outcom tonight jay game -home run -win pitcher eco god uwo                       |                 |
    | 11614 | sport       | lot run                                                                                              |                 |
    | 11660 | sport       | modest request find volum stuff                                                                      |                 |
    | 11664 | sport       | best first basemen mattingli best first baseman histori basebal alway ha alway                       |                 |
    | 11669 | sport       | jewish broadcast wa jewish basebal player let forget al michael believ miracl fame jim               |                 |
    | 11720 | sport       | fenway hi- doe anybodi know ticket info fenway thank eugenesrhim dartmouthedu                        |                 |
    | 11778 | sport       | jewish basebal player think add former first baseman mike epstein relat list ari                     |                 |
    | 11791 | sport       | best homerun say impress hr ever see came dave kingman hi infam moon-rak drive boell                 |                 |
    | 11795 | sport       | world seri stat doe anybodi els think ws stat becom part player career stat whi                      |                 |
    | 11802 | sport       | fenway hi- doe anybodi know ticket info fenway less  peopl  -   - steve                              |                 |
    | 11803 | sport       | ugliest stanc nt know think phil plantier ha ugliest stanc look like sit toilet brian tbo rosen      |                 |
    | 11804 | sport       | best homerun mike schmidt th onli mileston also th inning game- winner -john                         |                 |
    | 11866 | sport       | al stat stand forgot mention stat game  doug                                                         |                 |
    | 11884 | sport       | melido due back yanke plan activ melido perez hi  day dl today bring back thi weekend thank ani info |                 |
    | 11892 | sport       | jewish basebal player bo bilinski                                                                    |                 |
    | 11913 | sport       | slug percentag comput subject line say thank advanc pleas email chuck cygnuseidanlgov go cub         |                 |
    | 11927 | sport       | doe pitcher get save subject line say rule qualifi pitcher make save                                 |                 |
    | 12044 | sport       | wound redbird doe anyon know statu jeffri arocha                                                     |                 |
    | 12047 | sport       | hbp bb big-cat articl                                                                                |                 |
    | 12079 | sport       | all-tim best player b career mattingli                                                               |                 |
    | 12101 | sport       | yanke schedul someon thi net post yanke schedul need thi right away thank                            |                 |
    | 12116 | sport       | cubbi chicago cub mail list like join ani help appreci                                               |                 |
    | 12137 | sport       | stat question wonder whether offici mlb stat includ intent walk bb categori wenhsiang lin            |                 |
    | 12148 | sport       | giant win pennant giant win pennant giant win pennant gi ooop guess littl earli see octob            |                 |
    | 12155 | sport       | dodger newslett could somebodi pleas tell dodger newslett net subscrib thank joel                    |                 |
    | 12186 | sport       | red sox win st bosox  royal  wp clemen - lp appier - key hit mike greenwel  trippl base load         |                 |
    | 12192 | sport       | vega odd doe anyon list vega odd team make world seri appreci mail thank rickc corpsgicom            |                 |
    | 12193 | sport       |                                                                                                      |                 |
    | 12221 | sport       | david well ha david well land team yet think tiger anem pitch would grab thi guy pronto dc           |                 |
    | 12274 | sport       | sax ani news steve statu sinc lost start job would appreci thank gwyn                                |                 |
    | 12338 | sport       | cub expo roster question make room harkey cub sent shawn boski aaa                                   |                 |
    | 12399 | sport       | kevin roger hpcc                                                                                     |                 |
    | 12406 | sport       | fenway gif love see shea stadium gif -sean behind bag - vin sculli                                   |                 |
    | 12416 | sport       | nl vs al doug robert - ken hill nl mvp let go spo                                                    |                 |
    | 12422 | sport       | scott erickson doe anyon scoop scot erickson long go                                                 |                 |
    | 12425 | sport       | jewish basebal player dave kingman jewish bob comarow eisnerdecusorg                                 |                 |
    | 12443 | sport       | request al stat anyon al individu stat find k --                                                     |                 |
    | 12458 | sport       | twin game doe anyon know twin game broadcast good ole ame iowa thank                                 |                 |
    | 12506 | sport       | doe fred mcgriff padr becom free agent                                                               |                 |
    | 12532 | sport       | ryam - week nolan ryan ha torn cartlidg inhi right knee surgeri expect miss - week --                |                 |
    | 12535 | sport       | yanke fear rawley eastwick                                                                           |                 |
    | 12566 | sport       | jewish basebal player add steve rosenberg one-tim white sox reliev met system list greg              |                 |
    | 12583 | sport       | think go cri yanke lose ca nt believ thi howe ha era  improv key pitch great game screw              |                 |
    | 12589 | sport       | astro real think astro go place current first place - - road                                         |                 |
    | 12595 | sport       | oriol trivia bunker mcnalli later pappa estrada steve barber jay                                     |                 |
    | 12600 | sport       | jay game anyon know outcom tonight jay game -home run -win pitcher eco god uwo                       |                 |
    | 12608 | sport       | lot run                                                                                              |                 |
    | 12654 | sport       | modest request find volum stuff                                                                      |                 |
    | 12658 | sport       | best first basemen mattingli best first baseman histori basebal alway ha alway                       |                 |
    | 12663 | sport       | jewish broadcast wa jewish basebal player let forget al michael believ miracl fame jim               |                 |
    | 12714 | sport       | fenway hi- doe anybodi know ticket info fenway thank eugenesrhim dartmouthedu                        |                 |
    | 12772 | sport       | jewish basebal player think add former first baseman mike epstein relat list ari                     |                 |
    | 12785 | sport       | best homerun say impress hr ever see came dave kingman hi infam moon-rak drive boell                 |                 |
    | 12789 | sport       | world seri stat doe anybodi els think ws stat becom part player career stat whi                      |                 |
    | 12796 | sport       | fenway hi- doe anybodi know ticket info fenway less  peopl  -   - steve                              |                 |
    | 12797 | sport       | ugliest stanc nt know think phil plantier ha ugliest stanc look like sit toilet brian tbo rosen      |                 |
    | 12798 | sport       | best homerun mike schmidt th onli mileston also th inning game- winner -john                         |                 |
    | 12860 | sport       | al stat stand forgot mention stat game  doug                                                         |                 |
    | 12886 | sport       | jewish basebal player bo bilinski                                                                    |                 |
    | 12907 | sport       | slug percentag comput subject line say thank advanc pleas email chuck cygnuseidanlgov go cub         |                 |
    | 12921 | sport       | doe pitcher get save subject line say rule qualifi pitcher make save                                 |                 |
    | 13038 | sport       | wound redbird doe anyon know statu jeffri arocha                                                     |                 |
    | 13041 | sport       | hbp bb big-cat articl                                                                                |                 |
    | 13073 | sport       | all-tim best player b career mattingli                                                               |                 |
    | 13095 | sport       | yanke schedul someon thi net post yanke schedul need thi right away thank                            |                 |
    | 13110 | sport       | cubbi chicago cub mail list like join ani help appreci                                               |                 |
    | 13131 | sport       | stat question wonder whether offici mlb stat includ intent walk bb categori wenhsiang lin            |                 |
    | 13144 | sport       | ncaa final winner lake statemain final pleas post                                                    |                 |
    | 13163 | sport       | ncaa final winner ktgeiss miavxacsmuohioedu write lake statemain final pleas post main -             |                 |
    | 13191 | sport       | ncaa final winner ktgeiss miavxacsmuohi write lake statemain final pleas post main  lssu  terri      |                 |
    | 13241 | sport       | test                                                                                                 |                 |
    | 13255 | sport       | test test flame pleas bye                                                                            |                 |
    | 13325 | sport       | nhl team milwauke newsgroup                                                                          |                 |
    | 13332 | sport       | plu minu stat post   newsgroup                                                                       |                 |
    | 13396 | sport       | beat pittsburgh beat penguin crash team plane ryan                                                   |                 |
    | 13407 | sport       | help wc coverag europ vielen dank desper fan ps sweden vs finland finish - gothenburg th apr         |                 |
    | 13428 | sport       | plu minu stat post   newsgroup                                                                       |                 |
    | 13429 | sport       | onli test messag onli test messag                                                                    |                 |
    | 13438 | sport       | plu minu stat post   newsgroup                                                                       |                 |
    | 13473 | sport       | look boxscor look  boxscor ani nhl team person research someon help michel arsenault                 |                 |
    | 13521 | sport       | island sux need say                                                                                  |                 |
    | 13583 | sport       | beat pen ca nt good luck tri jim                                                                     |                 |
    | 13690 | sport       | stat doe anyon player stat game play april  mauro                                                    |                 |
    | 13703 | sport       | octopu detroit david                                                                                 |                 |
    | 13724 | sport       | leaf v wing agre rob shick suck big time thing last night game boston buffalo                        |                 |
    | 13732 | sport       | hockey guest spot articl                                                                             |                 |
    | 13780 | sport       | stat hello could someon tell could find faceoff stat  season later  think earli get thank lot jp     |                 |
    | 13810 | sport       | bruin could anyon post game summari sabres-bruin game                                                |                 |
    | 13811 | sport       | let go buffalo know never realli appreci befor                                                       |                 |
    | 13961 | sport       | nhl draft believ nhl draft june th weekend                                                           |                 |
    | 13987 | sport       | test pleas ignor - quot ohandley betsygsfcnasa - oh ohandley betsygsfcnasagov oh newsgroup           |                 |
    | 14103 | sport       | bruin fan good point - nt even ani recent post ulf secretli convinc respons bs - buffalo somehow     |                 |
    | 14108 | sport       | nhl letter want send letter nhl would send brett e ball                                              |                 |
    | 14142 | sport       | ncaa final winner lake statemain final pleas post                                                    |                 |
    | 14161 | sport       | ncaa final winner ktgeiss miavxacsmuohioedu write lake statemain final pleas post main -             |                 |
    | 14189 | sport       | ncaa final winner ktgeiss miavxacsmuohi write lake statemain final pleas post main  lssu  terri      |                 |
    | 14239 | sport       | test                                                                                                 |                 |
    | 14253 | sport       | test test flame pleas bye                                                                            |                 |
    | 14255 | sport       |                                                                                                      |                 |
    | 14324 | sport       | nhl team milwauke newsgroup                                                                          |                 |
    | 14331 | sport       | plu minu stat post   newsgroup                                                                       |                 |
    | 14395 | sport       | beat pittsburgh beat penguin crash team plane ryan                                                   |                 |
    | 14406 | sport       | help wc coverag europ vielen dank desper fan ps sweden vs finland finish - gothenburg th apr         |                 |
    | 14427 | sport       | plu minu stat post   newsgroup                                                                       |                 |
    | 14428 | sport       | onli test messag onli test messag                                                                    |                 |
    | 14437 | sport       | plu minu stat post   newsgroup                                                                       |                 |
    | 14472 | sport       | look boxscor look  boxscor ani nhl team person research someon help michel arsenault                 |                 |
    | 14520 | sport       | island sux need say                                                                                  |                 |
    | 14582 | sport       | beat pen ca nt good luck tri jim                                                                     |                 |
    | 14689 | sport       | stat doe anyon player stat game play april  mauro                                                    |                 |
    | 14702 | sport       | octopu detroit david                                                                                 |                 |
    | 14723 | sport       | leaf v wing agre rob shick suck big time thing last night game boston buffalo                        |                 |
    | 14731 | sport       | hockey guest spot articl                                                                             |                 |
    | 14779 | sport       | stat hello could someon tell could find faceoff stat  season later  think earli get thank lot jp     |                 |
    | 14809 | sport       | bruin could anyon post game summari sabres-bruin game                                                |                 |
    | 14810 | sport       | let go buffalo know never realli appreci befor                                                       |                 |
    | 14960 | sport       | nhl draft believ nhl draft june th weekend                                                           |                 |
    | 14986 | sport       | test pleas ignor - quot ohandley betsygsfcnasa - oh ohandley betsygsfcnasagov oh newsgroup           |                 |
    | 15102 | sport       | bruin fan good point - nt even ani recent post ulf secretli convinc respons bs - buffalo somehow     |                 |
    | 15107 | sport       | nhl letter want send letter nhl would send brett e ball                                              |                 |
    | 15955 | religion    | enviroleagu new altern scout unaccept bsa reason religi sexual prefer                                |                 |
    | 15964 | religion    | american evolut articl apr dcswarwickacuk simon dcswarwickacuk simon clippingdal write delet         |                 |
    | 16210 | religion    | moral constant wa biblic rape thi fray thread ha turn patent                                         |                 |
    | 16451 | religion    | thought sold sign brian -- next book charl manson lord lunat liar                                    |                 |
    | 16621 | religion    | free moral agenc articl  viceicotekcom bobb viceicotekcom robert beauchain write newsgroup           |                 |
    | 16624 | religion    | note bobbi articl apr daffycswiscedu mccullou snakecswiscedu mark mccullough write newsgroup         |                 |
    | 16657 | religion    | judg bobbi articl kmr pocwruedu kmr pocwruedu keith ryan write newsgroup                             |                 |
    | 16660 | religion    | thought articl  viceicotekcom bobb viceicotekcom robert beauchain write newsgroup                    |                 |
    | 16663 | religion    | judg bobbi articl apr ultbiscritedu snm ultbiscritedu sn mozumd write newsgroup                      |                 |
    | 16699 | religion    | thought christian excerpt netnew                                                                     |                 |
    | 16708 | religion    | thought bissda saturnwwcedu dan lawrenc bissel write first want start right say christian well thi   |                 |
    | 16747 | religion    | new member jcopelan nyxcsduedu one onli write welcom offici keeper list nicknam peopl known          |                 |
    | 17002 | religion    | christian moral dan schaertel dp nasakodakcom wrote sinc thi                                         |                 |
    | 17014 | religion    | amus atheist agnost articl timmbake mcl timmbak mclucsbedu bake timmon write newsgroup               |                 |
    | 17041 | religion    | societ accept behavior articl cqgmdl newscsouiucedu cobb alexialisuiucedu mike cobb write newsgroup  |                 |
    | 17080 | religion    | go hell articl apr nusccnussg cmtan issnussg tan chade meng - dan write newsgroup                    |                 |
    | 17107 | religion    | koresh god latest news seem koresh give onc finish write sequel bibl mathew                          |                 |
    | 17131 | religion    | go hell articl cszme cbnewsjcbattcom decay cbnewsjcbattcom deankaflowitz write newsgroup             |                 |
    | 17134 | religion    | death penalti gulf war long                                                                          |                 |
    | 17298 | religion    | request articl  viceicotekcom bobb viceicotekcom robert beauchain write newsgroup                    |                 |
    | 17301 | religion    | studi book mormon articl snx enkidumiccl agrino enkidumiccl andr grino brandt write newsgroup        |                 |
    | 17443 | religion    | bibl contradict would like list bibl contadict dispit free christian well vers bibl                  |                 |
    | 17449 | religion    | silli question x-tianiti articl pww- spac-at-riceedu pww spacsunriceedu peter walker write newsgroup |                 |
    | 17458 | religion    | go hell blashephem go hell believ god prepar etern damnat                                            |                 |
    | 17467 | religion    | islam clearer view articl bafci dbsturztu-bsd i dbsturztu-bsd benedikt rosenau write newsgroup       |                 |
    | 17656 | religion    | go hell jsn jeremi scott noonan                                                                      |                 |
    | 17662 | religion    | thought articl aa therosepdxcom alanolsen pfnzfidonetorg alan olsen write newsgroup                  |                 |
    | 17837 | religion    | atheism                                                                                              |                 |
    | 17903 | religion    | go hell thi hell ha nt anyon notic consensu realiti special case                                     |                 |
    | 18033 | religion    | free moral agenc see told wa atheist mytholog thank prove point bill                                 |                 |
    | 18147 | religion    | albert sabin br newsgroup                                                                            |                 |
    | 18150 | religion    | albert sabin br newsgroup                                                                            |                 |
    | 18156 | religion    | thought rh newsgroup                                                                                 |                 |
    | 18183 | religion    | nt innoc die without death penalti                                                                   |                 |
    | 18187 | religion    | ancient islam ritual                                                                                 |                 |
    | 18191 | religion    | polit atheist                                                                                        |                 |
    | 18200 | religion    | must creator mayb                                                                                    |                 |
    | 18203 | religion    | american evolut                                                                                      |                 |
    | 18208 | religion    | specul                                                                                               |                 |
    | 18211 | religion    | islam author women                                                                                   |                 |
    | 18219 | religion    | radic agnost                                                                                         |                 |
    | 18222 | religion    | maddi                                                                                                |                 |
    | 18225 | religion    | wrong right                                                                                          |                 |
    | 18232 | religion    | gospel date                                                                                          |                 |
    | 18237 | religion    | word advic                                                                                           |                 |
    | 18244 | religion    | omnipot wa specul                                                                                    |                 |
    | 18247 | religion    | islam marriag                                                                                        |                 |
    | 18256 | religion    | pompou ass                                                                                           |                 |
    | 18262 | religion    | concern god moral long                                                                               |                 |
    | 18269 | religion    | visit jehovah wit                                                                                    |                 |
    | 18275 | religion    | biblic rape                                                                                          |                 |
    | 18280 | religion    | visit jehovah wit good grief                                                                         |                 |
    | 18283 | religion    | vonnegutath                                                                                          |                 |
    | 18286 | religion    | yet rushdi islam law                                                                                 |                 |
    | 18289 | religion    | request support                                                                                      |                 |
    | 18292 | religion    | bill conner                                                                                          |                 |
    | 18295 | religion    | fluid vs liquid                                                                                      |                 |
    | 18300 | religion    | moral constant wa biblic rape                                                                        |                 |
    | 18305 | religion    | contradict                                                                                           |                 |
    | 18308 | religion    | period post charley challeng  addit                                                                  |                 |
    | 18311 | religion    | femin islam                                                                                          |                 |
    | 18314 | religion    | nonexist atheist                                                                                     |                 |
    | 18317 | religion    | ha read rushdi satan verses                                                                          |                 |
    | 18321 | religion    | death penalti wa polit athei                                                                         |                 |
    | 18324 | religion    | death penalti gulf war long                                                                          |                 |
    | 18328 | religion    | religion caus wa islam author women                                                                  |                 |
    | 18333 | religion    | inimit rushdi                                                                                        |                 |
    | 18334 | religion    | age reason wa ha read rushdi                                                                         |                 |
    | 18337 | religion    | strong weak atheism                                                                                  |                 |
    | 18340 | religion    | death penalti wa polit atheist                                                                       |                 |
    | 18343 | religion    | alleg deathb convers wa asimov stamp                                                                 |                 |
    | 18344 | religion    |                                                                                                      |                 |
    | 18345 | religion    |                                                                                                      |                 |
    | 18346 | religion    |                                                                                                      |                 |
    | 18347 | religion    |                                                                                                      |                 |
    | 18348 | religion    |                                                                                                      |                 |
    | 18349 | religion    |                                                                                                      |                 |
    | 18350 | religion    |                                                                                                      |                 |
    | 18351 | religion    |                                                                                                      |                 |
    | 18352 | religion    |                                                                                                      |                 |
    | 18353 | religion    |                                                                                                      |                 |
    | 18354 | religion    |                                                                                                      |                 |
    | 18355 | religion    |                                                                                                      |                 |
    | 18356 | religion    |                                                                                                      |                 |
    | 18357 | religion    |                                                                                                      |                 |
    | 18358 | religion    |                                                                                                      |                 |
    | 18359 | religion    |                                                                                                      |                 |
    | 18360 | religion    | nt innoc die without death penalti                                                                   |                 |
    | 18361 | religion    | ancient islam ritual                                                                                 |                 |
    | 18362 | religion    | polit atheist                                                                                        |                 |
    | 18363 | religion    | must creator mayb                                                                                    |                 |
    | 18364 | religion    | american evolut                                                                                      |                 |
    | 18365 | religion    | specul                                                                                               |                 |
    | 18366 | religion    |                                                                                                      |                 |
    | 18367 | religion    |                                                                                                      |                 |
    | 18368 | religion    |                                                                                                      |                 |
    | 18369 | religion    |                                                                                                      |                 |
    | 18370 | religion    | islam author women                                                                                   |                 |
    | 18371 | religion    | ancient islam ritual                                                                                 |                 |
    | 18372 | religion    |                                                                                                      |                 |
    | 18373 | religion    |                                                                                                      |                 |
    | 18374 | religion    |                                                                                                      |                 |
    | 18375 | religion    |                                                                                                      |                 |
    | 18376 | religion    |                                                                                                      |                 |
    | 18377 | religion    |                                                                                                      |                 |
    | 18378 | religion    |                                                                                                      |                 |
    | 18379 | religion    |                                                                                                      |                 |
    | 18380 | religion    |                                                                                                      |                 |
    | 18381 | religion    | radic agnost                                                                                         |                 |
    | 18382 | religion    | maddi                                                                                                |                 |
    | 18383 | religion    | wrong right                                                                                          |                 |
    | 18384 | religion    | polit atheist                                                                                        |                 |
    | 18385 | religion    | american evolut                                                                                      |                 |
    | 18386 | religion    |                                                                                                      |                 |
    | 18387 | religion    |                                                                                                      |                 |
    | 18388 | religion    |                                                                                                      |                 |
    | 18389 | religion    |                                                                                                      |                 |
    | 18390 | religion    |                                                                                                      |                 |
    | 18391 | religion    | gospel date                                                                                          |                 |
    | 18392 | religion    | word advic                                                                                           |                 |
    | 18393 | religion    |                                                                                                      |                 |
    | 18394 | religion    | islam author women                                                                                   |                 |
    | 18395 | religion    | islam author women                                                                                   |                 |
    | 18396 | religion    | omnipot wa specul                                                                                    |                 |
    | 18397 | religion    | islam marriag                                                                                        |                 |
    | 18398 | religion    | polit atheist                                                                                        |                 |
    | 18399 | religion    | polit atheist                                                                                        |                 |
    | 18400 | religion    | polit atheist                                                                                        |                 |
    | 18401 | religion    | pompou ass                                                                                           |                 |
    | 18402 | religion    | pompou ass                                                                                           |                 |
    | 18403 | religion    | concern god moral long                                                                               |                 |
    | 18404 | religion    |                                                                                                      |                 |
    | 18405 | religion    |                                                                                                      |                 |
    | 18406 | religion    |                                                                                                      |                 |
    | 18407 | religion    |                                                                                                      |                 |
    | 18408 | religion    |                                                                                                      |                 |
    | 18409 | religion    |                                                                                                      |                 |
    | 18410 | religion    |                                                                                                      |                 |
    | 18411 | religion    |                                                                                                      |                 |
    | 18412 | religion    |                                                                                                      |                 |
    | 18413 | religion    |                                                                                                      |                 |
    | 18414 | religion    |                                                                                                      |                 |
    | 18415 | religion    |                                                                                                      |                 |
    | 18416 | religion    |                                                                                                      |                 |
    | 18417 | religion    |                                                                                                      |                 |
    | 18418 | religion    |                                                                                                      |                 |
    | 18419 | religion    |                                                                                                      |                 |
    | 18420 | religion    |                                                                                                      |                 |
    | 18421 | religion    |                                                                                                      |                 |
    | 18422 | religion    |                                                                                                      |                 |
    | 18423 | religion    |                                                                                                      |                 |
    | 18424 | religion    |                                                                                                      |                 |
    | 18425 | religion    |                                                                                                      |                 |
    | 18426 | religion    | american evolut                                                                                      |                 |
    | 18427 | religion    |                                                                                                      |                 |
    | 18428 | religion    |                                                                                                      |                 |
    | 18429 | religion    |                                                                                                      |                 |
    | 18430 | religion    |                                                                                                      |                 |
    | 18431 | religion    |                                                                                                      |                 |
    | 18432 | religion    |                                                                                                      |                 |
    | 18433 | religion    |                                                                                                      |                 |
    | 18434 | religion    |                                                                                                      |                 |
    | 18435 | religion    |                                                                                                      |                 |
    | 18436 | religion    |                                                                                                      |                 |
    | 18437 | religion    |                                                                                                      |                 |
    | 18438 | religion    |                                                                                                      |                 |
    | 18439 | religion    |                                                                                                      |                 |
    | 18440 | religion    |                                                                                                      |                 |
    | 18441 | religion    | nt innoc die without death penalti                                                                   |                 |
    | 18442 | religion    | visit jehovah wit                                                                                    |                 |
    | 18443 | religion    | visit jehovah wit                                                                                    |                 |
    | 18444 | religion    |                                                                                                      |                 |
    | 18445 | religion    |                                                                                                      |                 |
    | 18446 | religion    |                                                                                                      |                 |
    | 18447 | religion    |                                                                                                      |                 |
    | 18448 | religion    |                                                                                                      |                 |
    | 18449 | religion    |                                                                                                      |                 |
    | 18450 | religion    |                                                                                                      |                 |
    | 18451 | religion    |                                                                                                      |                 |
    | 18452 | religion    |                                                                                                      |                 |
    | 18453 | religion    |                                                                                                      |                 |
    | 18454 | religion    |                                                                                                      |                 |
    | 18455 | religion    |                                                                                                      |                 |
    | 18456 | religion    |                                                                                                      |                 |
    | 18457 | religion    |                                                                                                      |                 |
    | 18458 | religion    |                                                                                                      |                 |
    | 18459 | religion    |                                                                                                      |                 |
    | 18460 | religion    |                                                                                                      |                 |
    | 18461 | religion    |                                                                                                      |                 |
    | 18462 | religion    |                                                                                                      |                 |
    | 18463 | religion    |                                                                                                      |                 |
    | 18464 | religion    |                                                                                                      |                 |
    | 18465 | religion    |                                                                                                      |                 |
    | 18466 | religion    |                                                                                                      |                 |
    | 18467 | religion    |                                                                                                      |                 |
    | 18468 | religion    |                                                                                                      |                 |
    | 18469 | religion    |                                                                                                      |                 |
    | 18470 | religion    |                                                                                                      |                 |
    | 18471 | religion    |                                                                                                      |                 |
    | 18472 | religion    |                                                                                                      |                 |
    | 18473 | religion    |                                                                                                      |                 |
    | 18474 | religion    |                                                                                                      |                 |
    | 18475 | religion    |                                                                                                      |                 |
    | 18476 | religion    |                                                                                                      |                 |
    | 18477 | religion    |                                                                                                      |                 |
    | 18478 | religion    | biblic rape                                                                                          |                 |
    | 18479 | religion    |                                                                                                      |                 |
    | 18480 | religion    |                                                                                                      |                 |
    | 18481 | religion    |                                                                                                      |                 |
    | 18482 | religion    |                                                                                                      |                 |
    | 18483 | religion    |                                                                                                      |                 |
    | 18484 | religion    |                                                                                                      |                 |
    | 18485 | religion    |                                                                                                      |                 |
    | 18486 | religion    |                                                                                                      |                 |
    | 18487 | religion    |                                                                                                      |                 |
    | 18488 | religion    |                                                                                                      |                 |
    | 18489 | religion    | polit atheist                                                                                        |                 |
    | 18490 | religion    |                                                                                                      |                 |
    | 18491 | religion    |                                                                                                      |                 |
    | 18492 | religion    |                                                                                                      |                 |
    | 18493 | religion    |                                                                                                      |                 |
    | 18494 | religion    |                                                                                                      |                 |
    | 18495 | religion    |                                                                                                      |                 |
    | 18496 | religion    |                                                                                                      |                 |
    | 18497 | religion    | islam author women                                                                                   |                 |
    | 18498 | religion    | visit jehovah wit good grief                                                                         |                 |
    | 18499 | religion    |                                                                                                      |                 |
    | 18500 | religion    | vonnegutath                                                                                          |                 |
    | 18501 | religion    | yet rushdi islam law                                                                                 |                 |
    | 18502 | religion    | request support                                                                                      |                 |
    | 18503 | religion    | bill conner                                                                                          |                 |
    | 18504 | religion    |                                                                                                      |                 |
    | 18505 | religion    |                                                                                                      |                 |
    | 18506 | religion    | fluid vs liquid                                                                                      |                 |
    | 18507 | religion    | gospel date                                                                                          |                 |
    | 18508 | religion    | moral constant wa biblic rape                                                                        |                 |
    | 18509 | religion    | gospel date                                                                                          |                 |
    | 18510 | religion    | contradict                                                                                           |                 |
    | 18511 | religion    |                                                                                                      |                 |
    | 18512 | religion    |                                                                                                      |                 |
    | 18513 | religion    |                                                                                                      |                 |
    | 18514 | religion    |                                                                                                      |                 |
    | 18515 | religion    |                                                                                                      |                 |
    | 18516 | religion    |                                                                                                      |                 |
    | 18517 | religion    |                                                                                                      |                 |
    | 18518 | religion    |                                                                                                      |                 |
    | 18519 | religion    | period post charley challeng  addit                                                                  |                 |
    | 18520 | religion    | polit atheist                                                                                        |                 |
    | 18521 | religion    |                                                                                                      |                 |
    | 18522 | religion    |                                                                                                      |                 |
    | 18523 | religion    |                                                                                                      |                 |
    | 18524 | religion    |                                                                                                      |                 |
    | 18525 | religion    |                                                                                                      |                 |
    | 18526 | religion    |                                                                                                      |                 |
    | 18527 | religion    |                                                                                                      |                 |
    | 18528 | religion    |                                                                                                      |                 |
    | 18529 | religion    |                                                                                                      |                 |
    | 18530 | religion    |                                                                                                      |                 |
    | 18531 | religion    |                                                                                                      |                 |
    | 18532 | religion    |                                                                                                      |                 |
    | 18533 | religion    |                                                                                                      |                 |
    | 18534 | religion    |                                                                                                      |                 |
    | 18535 | religion    |                                                                                                      |                 |
    | 18536 | religion    |                                                                                                      |                 |
    | 18537 | religion    |                                                                                                      |                 |
    | 18538 | religion    |                                                                                                      |                 |
    | 18539 | religion    |                                                                                                      |                 |
    | 18540 | religion    |                                                                                                      |                 |
    | 18541 | religion    |                                                                                                      |                 |
    | 18542 | religion    |                                                                                                      |                 |
    | 18543 | religion    |                                                                                                      |                 |
    | 18544 | religion    |                                                                                                      |                 |
    | 18545 | religion    |                                                                                                      |                 |
    | 18546 | religion    |                                                                                                      |                 |
    | 18547 | religion    |                                                                                                      |                 |
    | 18548 | religion    |                                                                                                      |                 |
    | 18549 | religion    |                                                                                                      |                 |
    | 18550 | religion    |                                                                                                      |                 |
    | 18551 | religion    |                                                                                                      |                 |
    | 18552 | religion    |                                                                                                      |                 |
    | 18553 | religion    |                                                                                                      |                 |
    | 18554 | religion    |                                                                                                      |                 |
    | 18555 | religion    |                                                                                                      |                 |
    | 18556 | religion    |                                                                                                      |                 |
    | 18557 | religion    |                                                                                                      |                 |
    | 18558 | religion    |                                                                                                      |                 |
    | 18559 | religion    |                                                                                                      |                 |
    | 18560 | religion    |                                                                                                      |                 |
    | 18561 | religion    |                                                                                                      |                 |
    | 18562 | religion    |                                                                                                      |                 |
    | 18563 | religion    |                                                                                                      |                 |
    | 18564 | religion    |                                                                                                      |                 |
    | 18565 | religion    |                                                                                                      |                 |
    | 18566 | religion    |                                                                                                      |                 |
    | 18567 | religion    |                                                                                                      |                 |
    | 18568 | religion    |                                                                                                      |                 |
    | 18569 | religion    |                                                                                                      |                 |
    | 18570 | religion    |                                                                                                      |                 |
    | 18571 | religion    |                                                                                                      |                 |
    | 18572 | religion    |                                                                                                      |                 |
    | 18573 | religion    |                                                                                                      |                 |
    | 18574 | religion    |                                                                                                      |                 |
    | 18575 | religion    |                                                                                                      |                 |
    | 18576 | religion    |                                                                                                      |                 |
    | 18577 | religion    | femin islam                                                                                          |                 |
    | 18578 | religion    |                                                                                                      |                 |
    | 18579 | religion    |                                                                                                      |                 |
    | 18580 | religion    |                                                                                                      |                 |
    | 18581 | religion    |                                                                                                      |                 |
    | 18582 | religion    |                                                                                                      |                 |
    | 18583 | religion    | nonexist atheist                                                                                     |                 |
    | 18584 | religion    |                                                                                                      |                 |
    | 18585 | religion    |                                                                                                      |                 |
    | 18586 | religion    |                                                                                                      |                 |
    | 18587 | religion    |                                                                                                      |                 |
    | 18588 | religion    |                                                                                                      |                 |
    | 18589 | religion    |                                                                                                      |                 |
    | 18590 | religion    |                                                                                                      |                 |
    | 18591 | religion    |                                                                                                      |                 |
    | 18592 | religion    |                                                                                                      |                 |
    | 18593 | religion    |                                                                                                      |                 |
    | 18594 | religion    |                                                                                                      |                 |
    | 18595 | religion    |                                                                                                      |                 |
    | 18596 | religion    |                                                                                                      |                 |
    | 18597 | religion    |                                                                                                      |                 |
    | 18598 | religion    |                                                                                                      |                 |
    | 18599 | religion    |                                                                                                      |                 |
    | 18600 | religion    |                                                                                                      |                 |
    | 18601 | religion    |                                                                                                      |                 |
    | 18602 | religion    |                                                                                                      |                 |
    | 18603 | religion    |                                                                                                      |                 |
    | 18604 | religion    |                                                                                                      |                 |
    | 18605 | religion    |                                                                                                      |                 |
    | 18606 | religion    |                                                                                                      |                 |
    | 18607 | religion    |                                                                                                      |                 |
    | 18608 | religion    |                                                                                                      |                 |
    | 18609 | religion    |                                                                                                      |                 |
    | 18610 | religion    |                                                                                                      |                 |
    | 18611 | religion    |                                                                                                      |                 |
    | 18612 | religion    |                                                                                                      |                 |
    | 18613 | religion    |                                                                                                      |                 |
    | 18614 | religion    |                                                                                                      |                 |
    | 18615 | religion    |                                                                                                      |                 |
    | 18616 | religion    |                                                                                                      |                 |
    | 18617 | religion    |                                                                                                      |                 |
    | 18618 | religion    |                                                                                                      |                 |
    | 18619 | religion    |                                                                                                      |                 |
    | 18620 | religion    |                                                                                                      |                 |
    | 18621 | religion    |                                                                                                      |                 |
    | 18622 | religion    |                                                                                                      |                 |
    | 18623 | religion    |                                                                                                      |                 |
    | 18624 | religion    |                                                                                                      |                 |
    | 18625 | religion    |                                                                                                      |                 |
    | 18626 | religion    |                                                                                                      |                 |
    | 18627 | religion    |                                                                                                      |                 |
    | 18628 | religion    |                                                                                                      |                 |
    | 18629 | religion    |                                                                                                      |                 |
    | 18630 | religion    |                                                                                                      |                 |
    | 18631 | religion    |                                                                                                      |                 |
    | 18632 | religion    |                                                                                                      |                 |
    | 18633 | religion    |                                                                                                      |                 |
    | 18634 | religion    |                                                                                                      |                 |
    | 18635 | religion    |                                                                                                      |                 |
    | 18636 | religion    |                                                                                                      |                 |
    | 18637 | religion    |                                                                                                      |                 |
    | 18638 | religion    |                                                                                                      |                 |
    | 18639 | religion    |                                                                                                      |                 |
    | 18640 | religion    |                                                                                                      |                 |
    | 18641 | religion    |                                                                                                      |                 |
    | 18642 | religion    |                                                                                                      |                 |
    | 18643 | religion    |                                                                                                      |                 |
    | 18644 | religion    |                                                                                                      |                 |
    | 18645 | religion    |                                                                                                      |                 |
    | 18646 | religion    |                                                                                                      |                 |
    | 18647 | religion    |                                                                                                      |                 |
    | 18648 | religion    |                                                                                                      |                 |
    | 18649 | religion    |                                                                                                      |                 |
    | 18650 | religion    |                                                                                                      |                 |
    | 18651 | religion    |                                                                                                      |                 |
    | 18652 | religion    |                                                                                                      |                 |
    | 18653 | religion    |                                                                                                      |                 |
    | 18654 | religion    |                                                                                                      |                 |
    | 18655 | religion    |                                                                                                      |                 |
    | 18656 | religion    |                                                                                                      |                 |
    | 18657 | religion    |                                                                                                      |                 |
    | 18658 | religion    |                                                                                                      |                 |
    | 18659 | religion    |                                                                                                      |                 |
    | 18660 | religion    |                                                                                                      |                 |
    | 18661 | religion    |                                                                                                      |                 |
    | 18662 | religion    |                                                                                                      |                 |
    | 18663 | religion    |                                                                                                      |                 |
    | 18664 | religion    |                                                                                                      |                 |
    | 18665 | religion    |                                                                                                      |                 |
    | 18666 | religion    |                                                                                                      |                 |
    | 18667 | religion    |                                                                                                      |                 |
    | 18668 | religion    |                                                                                                      |                 |
    | 18669 | religion    |                                                                                                      |                 |
    | 18670 | religion    |                                                                                                      |                 |
    | 18671 | religion    |                                                                                                      |                 |
    | 18672 | religion    |                                                                                                      |                 |
    | 18673 | religion    |                                                                                                      |                 |
    | 18674 | religion    |                                                                                                      |                 |
    | 18675 | religion    |                                                                                                      |                 |
    | 18676 | religion    |                                                                                                      |                 |
    | 18677 | religion    |                                                                                                      |                 |
    | 18678 | religion    |                                                                                                      |                 |
    | 18679 | religion    |                                                                                                      |                 |
    | 18680 | religion    |                                                                                                      |                 |
    | 18681 | religion    |                                                                                                      |                 |
    | 18682 | religion    |                                                                                                      |                 |
    | 18683 | religion    |                                                                                                      |                 |
    | 18684 | religion    |                                                                                                      |                 |
    | 18685 | religion    |                                                                                                      |                 |
    | 18686 | religion    |                                                                                                      |                 |
    | 18687 | religion    |                                                                                                      |                 |
    | 18688 | religion    |                                                                                                      |                 |
    | 18689 | religion    |                                                                                                      |                 |
    | 18690 | religion    |                                                                                                      |                 |
    | 18691 | religion    |                                                                                                      |                 |
    | 18692 | religion    |                                                                                                      |                 |
    | 18693 | religion    |                                                                                                      |                 |
    | 18694 | religion    |                                                                                                      |                 |
    | 18695 | religion    |                                                                                                      |                 |
    | 18696 | religion    |                                                                                                      |                 |
    | 18697 | religion    |                                                                                                      |                 |
    | 18698 | religion    |                                                                                                      |                 |
    | 18699 | religion    |                                                                                                      |                 |
    | 18700 | religion    |                                                                                                      |                 |
    | 18701 | religion    |                                                                                                      |                 |
    | 18702 | religion    |                                                                                                      |                 |
    | 18703 | religion    |                                                                                                      |                 |
    | 18704 | religion    |                                                                                                      |                 |
    | 18705 | religion    |                                                                                                      |                 |
    | 18706 | religion    |                                                                                                      |                 |
    | 18707 | religion    |                                                                                                      |                 |
    | 18708 | religion    |                                                                                                      |                 |
    | 18709 | religion    |                                                                                                      |                 |
    | 18710 | religion    |                                                                                                      |                 |
    | 18711 | religion    |                                                                                                      |                 |
    | 18712 | religion    |                                                                                                      |                 |
    | 18713 | religion    |                                                                                                      |                 |
    | 18714 | religion    |                                                                                                      |                 |
    | 18715 | religion    |                                                                                                      |                 |
    | 18716 | religion    |                                                                                                      |                 |
    | 18717 | religion    |                                                                                                      |                 |
    | 18718 | religion    |                                                                                                      |                 |
    | 18719 | religion    |                                                                                                      |                 |
    | 18720 | religion    |                                                                                                      |                 |
    | 18721 | religion    |                                                                                                      |                 |
    | 18722 | religion    |                                                                                                      |                 |
    | 18723 | religion    |                                                                                                      |                 |
    | 18724 | religion    |                                                                                                      |                 |
    | 18725 | religion    |                                                                                                      |                 |
    | 18726 | religion    |                                                                                                      |                 |
    | 18727 | religion    |                                                                                                      |                 |
    | 18728 | religion    |                                                                                                      |                 |
    | 18729 | religion    |                                                                                                      |                 |
    | 18730 | religion    |                                                                                                      |                 |
    | 18731 | religion    |                                                                                                      |                 |
    | 18732 | religion    |                                                                                                      |                 |
    | 18733 | religion    |                                                                                                      |                 |
    | 18734 | religion    |                                                                                                      |                 |
    | 18735 | religion    |                                                                                                      |                 |
    | 18736 | religion    |                                                                                                      |                 |
    | 18737 | religion    |                                                                                                      |                 |
    | 18738 | religion    |                                                                                                      |                 |
    | 18739 | religion    |                                                                                                      |                 |
    | 18740 | religion    |                                                                                                      |                 |
    | 18741 | religion    |                                                                                                      |                 |
    | 18742 | religion    |                                                                                                      |                 |
    | 18743 | religion    |                                                                                                      |                 |
    | 18744 | religion    |                                                                                                      |                 |
    | 18745 | religion    |                                                                                                      |                 |
    | 18746 | religion    |                                                                                                      |                 |
    | 18747 | religion    |                                                                                                      |                 |
    | 18748 | religion    |                                                                                                      |                 |
    | 18749 | religion    |                                                                                                      |                 |
    | 18750 | religion    |                                                                                                      |                 |
    | 18751 | religion    |                                                                                                      |                 |
    | 18752 | religion    |                                                                                                      |                 |
    | 18753 | religion    |                                                                                                      |                 |
    | 18754 | religion    |                                                                                                      |                 |
    | 18755 | religion    |                                                                                                      |                 |
    | 18756 | religion    |                                                                                                      |                 |
    | 18757 | religion    |                                                                                                      |                 |
    | 18758 | religion    |                                                                                                      |                 |
    | 18759 | religion    |                                                                                                      |                 |
    | 18760 | religion    |                                                                                                      |                 |
    | 18761 | religion    |                                                                                                      |                 |
    | 18762 | religion    |                                                                                                      |                 |
    | 18763 | religion    |                                                                                                      |                 |
    | 18764 | religion    |                                                                                                      |                 |
    | 18765 | religion    |                                                                                                      |                 |
    | 18766 | religion    |                                                                                                      |                 |
    | 18767 | religion    |                                                                                                      |                 |
    | 18768 | religion    |                                                                                                      |                 |
    | 18769 | religion    |                                                                                                      |                 |
    | 18770 | religion    |                                                                                                      |                 |
    | 18771 | religion    |                                                                                                      |                 |
    | 18772 | religion    |                                                                                                      |                 |
    | 18773 | religion    |                                                                                                      |                 |
    | 18774 | religion    |                                                                                                      |                 |
    | 18775 | religion    |                                                                                                      |                 |
    | 18776 | religion    |                                                                                                      |                 |
    | 18777 | religion    |                                                                                                      |                 |
    | 18778 | religion    |                                                                                                      |                 |
    | 18779 | religion    |                                                                                                      |                 |
    | 18780 | religion    |                                                                                                      |                 |
    | 18781 | religion    |                                                                                                      |                 |
    | 18782 | religion    |                                                                                                      |                 |
    | 18783 | religion    |                                                                                                      |                 |
    | 18784 | religion    |                                                                                                      |                 |
    | 18785 | religion    |                                                                                                      |                 |
    | 18786 | religion    |                                                                                                      |                 |
    | 18787 | religion    |                                                                                                      |                 |
    | 18788 | religion    |                                                                                                      |                 |
    | 18789 | religion    |                                                                                                      |                 |
    | 18790 | religion    |                                                                                                      |                 |
    | 18791 | religion    |                                                                                                      |                 |
    | 18792 | religion    |                                                                                                      |                 |
    | 18793 | religion    |                                                                                                      |                 |
    | 18794 | religion    |                                                                                                      |                 |
    | 18795 | religion    |                                                                                                      |                 |
    | 18796 | religion    |                                                                                                      |                 |
    | 18797 | religion    |                                                                                                      |                 |
    | 18798 | religion    |                                                                                                      |                 |
    | 18799 | religion    |                                                                                                      |                 |
    | 18800 | religion    |                                                                                                      |                 |
    | 18801 | religion    |                                                                                                      |                 |
    | 18802 | religion    |                                                                                                      |                 |
    | 18803 | religion    |                                                                                                      |                 |
    | 18804 | religion    |                                                                                                      |                 |
    | 18805 | religion    | ha read rushdi satan verses                                                                          |                 |
    | 18806 | religion    |                                                                                                      |                 |
    | 18807 | religion    |                                                                                                      |                 |
    | 18808 | religion    |                                                                                                      |                 |
    | 18809 | religion    |                                                                                                      |                 |
    | 18810 | religion    |                                                                                                      |                 |
    | 18811 | religion    |                                                                                                      |                 |
    | 18812 | religion    |                                                                                                      |                 |
    | 18813 | religion    |                                                                                                      |                 |
    | 18814 | religion    |                                                                                                      |                 |
    | 18815 | religion    |                                                                                                      |                 |
    | 18816 | religion    |                                                                                                      |                 |
    | 18817 | religion    |                                                                                                      |                 |
    | 18818 | religion    |                                                                                                      |                 |
    | 18819 | religion    |                                                                                                      |                 |
    | 18820 | religion    |                                                                                                      |                 |
    | 18821 | religion    |                                                                                                      |                 |
    | 18822 | religion    |                                                                                                      |                 |
    | 18823 | religion    |                                                                                                      |                 |
    | 18824 | religion    |                                                                                                      |                 |
    | 18825 | religion    |                                                                                                      |                 |
    | 18826 | religion    |                                                                                                      |                 |
    | 18827 | religion    |                                                                                                      |                 |
    | 18828 | religion    |                                                                                                      |                 |
    | 18829 | religion    |                                                                                                      |                 |
    | 18830 | religion    |                                                                                                      |                 |
    | 18831 | religion    |                                                                                                      |                 |
    | 18832 | religion    |                                                                                                      |                 |
    | 18833 | religion    |                                                                                                      |                 |
    | 18834 | religion    |                                                                                                      |                 |
    | 18835 | religion    |                                                                                                      |                 |
    | 18836 | religion    |                                                                                                      |                 |
    | 18837 | religion    |                                                                                                      |                 |
    | 18838 | religion    | death penalti wa polit athei                                                                         |                 |
    | 18839 | religion    |                                                                                                      |                 |
    | 18840 | religion    |                                                                                                      |                 |
    | 18841 | religion    |                                                                                                      |                 |
    | 18842 | religion    |                                                                                                      |                 |
    | 18843 | religion    |                                                                                                      |                 |
    | 18844 | religion    |                                                                                                      |                 |
    | 18845 | religion    |                                                                                                      |                 |
    | 18846 | religion    |                                                                                                      |                 |
    | 18847 | religion    |                                                                                                      |                 |
    | 18848 | religion    |                                                                                                      |                 |
    | 18849 | religion    |                                                                                                      |                 |
    | 18850 | religion    |                                                                                                      |                 |
    | 18851 | religion    |                                                                                                      |                 |
    | 18852 | religion    |                                                                                                      |                 |
    | 18853 | religion    |                                                                                                      |                 |
    | 18854 | religion    |                                                                                                      |                 |
    | 18855 | religion    |                                                                                                      |                 |
    | 18856 | religion    |                                                                                                      |                 |
    | 18857 | religion    |                                                                                                      |                 |
    | 18858 | religion    |                                                                                                      |                 |
    | 18859 | religion    |                                                                                                      |                 |
    | 18860 | religion    |                                                                                                      |                 |
    | 18861 | religion    |                                                                                                      |                 |
    | 18862 | religion    |                                                                                                      |                 |
    | 18863 | religion    |                                                                                                      |                 |
    | 18864 | religion    |                                                                                                      |                 |
    | 18865 | religion    |                                                                                                      |                 |
    | 18866 | religion    |                                                                                                      |                 |
    | 18867 | religion    |                                                                                                      |                 |
    | 18868 | religion    | death penalti gulf war long                                                                          |                 |
    | 18869 | religion    |                                                                                                      |                 |
    | 18870 | religion    |                                                                                                      |                 |
    | 18871 | religion    |                                                                                                      |                 |
    | 18872 | religion    |                                                                                                      |                 |
    | 18873 | religion    |                                                                                                      |                 |
    | 18874 | religion    |                                                                                                      |                 |
    | 18875 | religion    |                                                                                                      |                 |
    | 18876 | religion    |                                                                                                      |                 |
    | 18877 | religion    |                                                                                                      |                 |
    | 18878 | religion    |                                                                                                      |                 |
    | 18879 | religion    |                                                                                                      |                 |
    | 18880 | religion    |                                                                                                      |                 |
    | 18881 | religion    |                                                                                                      |                 |
    | 18882 | religion    |                                                                                                      |                 |
    | 18883 | religion    |                                                                                                      |                 |
    | 18884 | religion    |                                                                                                      |                 |
    | 18885 | religion    |                                                                                                      |                 |
    | 18886 | religion    |                                                                                                      |                 |
    | 18887 | religion    |                                                                                                      |                 |
    | 18888 | religion    |                                                                                                      |                 |
    | 18889 | religion    |                                                                                                      |                 |
    | 18890 | religion    |                                                                                                      |                 |
    | 18891 | religion    |                                                                                                      |                 |
    | 18892 | religion    |                                                                                                      |                 |
    | 18893 | religion    |                                                                                                      |                 |
    | 18894 | religion    |                                                                                                      |                 |
    | 18895 | religion    |                                                                                                      |                 |
    | 18896 | religion    |                                                                                                      |                 |
    | 18897 | religion    |                                                                                                      |                 |
    | 18898 | religion    |                                                                                                      |                 |
    | 18899 | religion    |                                                                                                      |                 |
    | 18900 | religion    |                                                                                                      |                 |
    | 18901 | religion    |                                                                                                      |                 |
    | 18902 | religion    |                                                                                                      |                 |
    | 18903 | religion    |                                                                                                      |                 |
    | 18904 | religion    |                                                                                                      |                 |
    | 18905 | religion    |                                                                                                      |                 |
    | 18906 | religion    |                                                                                                      |                 |
    | 18907 | religion    |                                                                                                      |                 |
    | 18908 | religion    |                                                                                                      |                 |
    | 18909 | religion    |                                                                                                      |                 |
    | 18910 | religion    |                                                                                                      |                 |
    | 18911 | religion    |                                                                                                      |                 |
    | 18912 | religion    |                                                                                                      |                 |
    | 18913 | religion    |                                                                                                      |                 |
    | 18914 | religion    |                                                                                                      |                 |
    | 18915 | religion    |                                                                                                      |                 |
    | 18916 | religion    |                                                                                                      |                 |
    | 18917 | religion    |                                                                                                      |                 |
    | 18918 | religion    |                                                                                                      |                 |
    | 18919 | religion    |                                                                                                      |                 |
    | 18920 | religion    |                                                                                                      |                 |
    | 18921 | religion    |                                                                                                      |                 |
    | 18922 | religion    |                                                                                                      |                 |
    | 18923 | religion    |                                                                                                      |                 |
    | 18924 | religion    |                                                                                                      |                 |
    | 18925 | religion    |                                                                                                      |                 |
    | 18926 | religion    |                                                                                                      |                 |
    | 18927 | religion    |                                                                                                      |                 |
    | 18928 | religion    |                                                                                                      |                 |
    | 18929 | religion    |                                                                                                      |                 |
    | 18930 | religion    |                                                                                                      |                 |
    | 18931 | religion    |                                                                                                      |                 |
    | 18932 | religion    |                                                                                                      |                 |
    | 18933 | religion    |                                                                                                      |                 |
    | 18934 | religion    |                                                                                                      |                 |
    | 18935 | religion    |                                                                                                      |                 |
    | 18936 | religion    |                                                                                                      |                 |
    | 18937 | religion    |                                                                                                      |                 |
    | 18938 | religion    |                                                                                                      |                 |
    | 18939 | religion    |                                                                                                      |                 |
    | 18940 | religion    |                                                                                                      |                 |
    | 18941 | religion    |                                                                                                      |                 |
    | 18942 | religion    |                                                                                                      |                 |
    | 18943 | religion    |                                                                                                      |                 |
    | 18944 | religion    |                                                                                                      |                 |
    | 18945 | religion    |                                                                                                      |                 |
    | 18946 | religion    |                                                                                                      |                 |
    | 18947 | religion    |                                                                                                      |                 |
    | 18948 | religion    |                                                                                                      |                 |
    | 18949 | religion    |                                                                                                      |                 |
    | 18950 | religion    |                                                                                                      |                 |
    | 18951 | religion    |                                                                                                      |                 |
    | 18952 | religion    |                                                                                                      |                 |
    | 18953 | religion    |                                                                                                      |                 |
    | 18954 | religion    |                                                                                                      |                 |
    | 18955 | religion    |                                                                                                      |                 |
    | 18956 | religion    |                                                                                                      |                 |
    | 18957 | religion    |                                                                                                      |                 |
    | 18958 | religion    |                                                                                                      |                 |
    | 18959 | religion    |                                                                                                      |                 |
    | 18960 | religion    |                                                                                                      |                 |
    | 18961 | religion    |                                                                                                      |                 |
    | 18962 | religion    |                                                                                                      |                 |
    | 18963 | religion    |                                                                                                      |                 |
    | 18964 | religion    |                                                                                                      |                 |
    | 18965 | religion    |                                                                                                      |                 |
    | 18966 | religion    |                                                                                                      |                 |
    | 18967 | religion    |                                                                                                      |                 |
    | 18968 | religion    |                                                                                                      |                 |
    | 18969 | religion    |                                                                                                      |                 |
    | 18970 | religion    |                                                                                                      |                 |
    | 18971 | religion    |                                                                                                      |                 |
    | 18972 | religion    |                                                                                                      |                 |
    | 18973 | religion    |                                                                                                      |                 |
    | 18974 | religion    |                                                                                                      |                 |
    | 18975 | religion    |                                                                                                      |                 |
    | 18976 | religion    |                                                                                                      |                 |
    | 18977 | religion    |                                                                                                      |                 |
    | 18978 | religion    |                                                                                                      |                 |
    | 18979 | religion    |                                                                                                      |                 |
    | 18980 | religion    |                                                                                                      |                 |
    | 18981 | religion    |                                                                                                      |                 |
    | 18982 | religion    |                                                                                                      |                 |
    | 18983 | religion    |                                                                                                      |                 |
    | 18984 | religion    |                                                                                                      |                 |
    | 18985 | religion    |                                                                                                      |                 |
    | 18986 | religion    |                                                                                                      |                 |
    | 18987 | religion    |                                                                                                      |                 |
    | 18988 | religion    |                                                                                                      |                 |
    | 18989 | religion    |                                                                                                      |                 |
    | 18990 | religion    | religion caus wa islam author women                                                                  |                 |
    | 18991 | religion    |                                                                                                      |                 |
    | 18992 | religion    |                                                                                                      |                 |
    | 18993 | religion    |                                                                                                      |                 |
    | 18994 | religion    | ha read rushdi satan verses                                                                          |                 |
    | 18995 | religion    | inimit rushdi                                                                                        |                 |
    | 18996 | religion    |                                                                                                      |                 |
    | 18997 | religion    |                                                                                                      |                 |
    | 18998 | religion    |                                                                                                      |                 |
    | 18999 | religion    |                                                                                                      |                 |
    | 19000 | religion    |                                                                                                      |                 |
    | 19001 | religion    |                                                                                                      |                 |
    | 19002 | religion    |                                                                                                      |                 |
    | 19003 | religion    |                                                                                                      |                 |
    | 19004 | religion    |                                                                                                      |                 |
    | 19005 | religion    |                                                                                                      |                 |
    | 19006 | religion    |                                                                                                      |                 |
    | 19007 | religion    |                                                                                                      |                 |
    | 19008 | religion    |                                                                                                      |                 |
    | 19009 | religion    |                                                                                                      |                 |
    | 19010 | religion    |                                                                                                      |                 |
    | 19011 | religion    |                                                                                                      |                 |
    | 19012 | religion    |                                                                                                      |                 |
    | 19013 | religion    |                                                                                                      |                 |
    | 19014 | religion    |                                                                                                      |                 |
    | 19015 | religion    |                                                                                                      |                 |
    | 19016 | religion    |                                                                                                      |                 |
    | 19017 | religion    |                                                                                                      |                 |
    | 19018 | religion    |                                                                                                      |                 |
    | 19019 | religion    |                                                                                                      |                 |
    | 19020 | religion    |                                                                                                      |                 |
    | 19021 | religion    |                                                                                                      |                 |
    | 19022 | religion    |                                                                                                      |                 |
    | 19023 | religion    |                                                                                                      |                 |
    | 19024 | religion    | age reason wa ha read rushdi                                                                         |                 |
    | 19025 | religion    | strong weak atheism                                                                                  |                 |
    | 19026 | religion    |                                                                                                      |                 |
    | 19027 | religion    |                                                                                                      |                 |
    | 19028 | religion    |                                                                                                      |                 |
    | 19029 | religion    |                                                                                                      |                 |
    | 19030 | religion    |                                                                                                      |                 |
    | 19031 | religion    |                                                                                                      |                 |
    | 19032 | religion    |                                                                                                      |                 |
    | 19033 | religion    |                                                                                                      |                 |
    | 19034 | religion    |                                                                                                      |                 |
    | 19035 | religion    |                                                                                                      |                 |
    | 19036 | religion    |                                                                                                      |                 |
    | 19037 | religion    |                                                                                                      |                 |
    | 19038 | religion    |                                                                                                      |                 |
    | 19039 | religion    |                                                                                                      |                 |
    | 19040 | religion    |                                                                                                      |                 |
    | 19041 | religion    |                                                                                                      |                 |
    | 19042 | religion    |                                                                                                      |                 |
    | 19043 | religion    |                                                                                                      |                 |
    | 19044 | religion    |                                                                                                      |                 |
    | 19045 | religion    |                                                                                                      |                 |
    | 19046 | religion    |                                                                                                      |                 |
    | 19047 | religion    |                                                                                                      |                 |
    | 19048 | religion    |                                                                                                      |                 |
    | 19049 | religion    |                                                                                                      |                 |
    | 19050 | religion    |                                                                                                      |                 |
    | 19051 | religion    |                                                                                                      |                 |
    | 19052 | religion    |                                                                                                      |                 |
    | 19053 | religion    |                                                                                                      |                 |
    | 19054 | religion    |                                                                                                      |                 |
    | 19055 | religion    |                                                                                                      |                 |
    | 19056 | religion    |                                                                                                      |                 |
    | 19057 | religion    |                                                                                                      |                 |
    | 19058 | religion    |                                                                                                      |                 |
    | 19059 | religion    |                                                                                                      |                 |
    | 19060 | religion    |                                                                                                      |                 |
    | 19061 | religion    |                                                                                                      |                 |
    | 19062 | religion    |                                                                                                      |                 |
    | 19063 | religion    |                                                                                                      |                 |
    | 19064 | religion    |                                                                                                      |                 |
    | 19065 | religion    |                                                                                                      |                 |
    | 19066 | religion    |                                                                                                      |                 |
    | 19067 | religion    |                                                                                                      |                 |
    | 19068 | religion    |                                                                                                      |                 |
    | 19069 | religion    |                                                                                                      |                 |
    | 19070 | religion    |                                                                                                      |                 |
    | 19071 | religion    |                                                                                                      |                 |
    | 19072 | religion    |                                                                                                      |                 |
    | 19073 | religion    |                                                                                                      |                 |
    | 19074 | religion    |                                                                                                      |                 |
    | 19075 | religion    |                                                                                                      |                 |
    | 19076 | religion    |                                                                                                      |                 |
    | 19077 | religion    |                                                                                                      |                 |
    | 19078 | religion    |                                                                                                      |                 |
    | 19079 | religion    |                                                                                                      |                 |
    | 19080 | religion    |                                                                                                      |                 |
    | 19081 | religion    |                                                                                                      |                 |
    | 19082 | religion    |                                                                                                      |                 |
    | 19083 | religion    |                                                                                                      |                 |
    | 19084 | religion    |                                                                                                      |                 |
    | 19085 | religion    |                                                                                                      |                 |
    | 19086 | religion    | death penalti wa polit atheist                                                                       |                 |
    | 19087 | religion    |                                                                                                      |                 |
    | 19088 | religion    |                                                                                                      |                 |
    | 19089 | religion    |                                                                                                      |                 |
    | 19090 | religion    |                                                                                                      |                 |
    | 19091 | religion    |                                                                                                      |                 |
    | 19092 | religion    |                                                                                                      |                 |
    | 19093 | religion    |                                                                                                      |                 |
    | 19094 | religion    |                                                                                                      |                 |
    | 19095 | religion    |                                                                                                      |                 |
    | 19096 | religion    |                                                                                                      |                 |
    | 19097 | religion    |                                                                                                      |                 |
    | 19098 | religion    |                                                                                                      |                 |
    | 19099 | religion    | death penalti gulf war long                                                                          |                 |
    | 19100 | religion    |                                                                                                      |                 |
    | 19101 | religion    |                                                                                                      |                 |
    | 19102 | religion    |                                                                                                      |                 |
    | 19103 | religion    |                                                                                                      |                 |
    | 19104 | religion    |                                                                                                      |                 |
    | 19105 | religion    |                                                                                                      |                 |
    | 19106 | religion    |                                                                                                      |                 |
    | 19107 | religion    |                                                                                                      |                 |
    | 19108 | religion    |                                                                                                      |                 |
    | 19109 | religion    |                                                                                                      |                 |
    | 19110 | religion    |                                                                                                      |                 |
    | 19111 | religion    |                                                                                                      |                 |
    | 19112 | religion    |                                                                                                      |                 |
    | 19113 | religion    |                                                                                                      |                 |
    | 19114 | religion    |                                                                                                      |                 |
    | 19115 | religion    |                                                                                                      |                 |
    | 19116 | religion    |                                                                                                      |                 |
    | 19117 | religion    |                                                                                                      |                 |
    | 19118 | religion    |                                                                                                      |                 |
    | 19119 | religion    |                                                                                                      |                 |
    | 19120 | religion    |                                                                                                      |                 |
    | 19121 | religion    |                                                                                                      |                 |
    | 19122 | religion    |                                                                                                      |                 |
    | 19123 | religion    |                                                                                                      |                 |
    | 19124 | religion    |                                                                                                      |                 |
    | 19125 | religion    |                                                                                                      |                 |
    | 19126 | religion    |                                                                                                      |                 |
    | 19127 | religion    |                                                                                                      |                 |
    | 19128 | religion    |                                                                                                      |                 |
    | 19129 | religion    |                                                                                                      |                 |
    | 19130 | religion    |                                                                                                      |                 |
    | 19131 | religion    |                                                                                                      |                 |
    | 19132 | religion    |                                                                                                      |                 |
    | 19133 | religion    |                                                                                                      |                 |
    | 19134 | religion    |                                                                                                      |                 |
    | 19135 | religion    |                                                                                                      |                 |
    | 19136 | religion    |                                                                                                      |                 |
    | 19137 | religion    |                                                                                                      |                 |
    | 19138 | religion    |                                                                                                      |                 |
    | 19139 | religion    |                                                                                                      |                 |
    | 19140 | religion    |                                                                                                      |                 |
    | 19141 | religion    |                                                                                                      |                 |
    | 19142 | religion    | alleg deathb convers wa asimov stamp                                                                 |                 |
    | 19147 | religion    | daili vers therefor whoever humbl like thi child greatest kingdom heaven matthew                     |                 |
    | 19156 | religion    | daili vers devot one anoth brotherli love honor one anoth abov yourselv roman                        |                 |
    | 19175 | religion    | doug sturm anyon familiar doug sturm pleas post think                                                |                 |
    | 19196 | religion    | salvat deed anoth guess salvat riddl would save joe fisher                                           |                 |
    | 19222 | religion    | prophet warn new york citi -- note repli messag                                                      |                 |
    | 19238 | religion    | propheci nyc marka traviscsdharriscom mark ashley write                                              |                 |
    | 19267 | religion    | end discuss easter close thi onc befor real tonight post                                             |                 |
    | 19296 | religion    | mission aviat fellowship hi doe anyon know anyth thi group ani info would appreci thank              |                 |
    | 19302 | religion    | daili vers overcom inherit thi hi god son revel                                                      |                 |
    | 19322 | religion    | daili vers much better get wisdom gold choos understand rather silver proverb                        |                 |
    | 19334 | religion    | daili vers whoever doe father heaven brother sister mother matthew                                   |                 |
    | 19411 | religion    | daili vers abov love deepli becaus love cover multitud sin ipet                                      |                 |
    | 19420 | religion    | love europ ani reader src go love europ congress germani thi juli -- michael davi csmcd brunelacuk   |                 |
    | 19422 | religion    | daili vers someon say faith deed show faith without deed show faith jame                             |                 |
    | 19482 | religion    | atheist view christian wa accept jeesu heart -                                                       |                 |
    | 19558 | religion    | sabbath admiss of follow thi thread talkreligion                                                     |                 |
    | 19559 | religion    | none satan                                                                                           |                 |
    | 19605 | religion    | daili vers let us becom weari good proper time reap harvest give galatian                            |                 |
    | 19648 | religion    | daili vers god peac soon crush satan feet grace lord jesu roman                                      |                 |
    | 19660 | religion    | wa immacul concept note cross-post actual email thi bitlistservcathol main post goe                  |                 |
    | 19709 | religion    | translat version bibl consid accur translat                                                          |                 |
    | 19780 | religion    | daili vers peac ha made two one ha destroy barrier divid wall hostil ephesian                        |                 |
    | 19877 | religion    | daili vers receiv power holi spirit come wit jerusalem judea samaria end earth act                   |                 |
    | 19944 | religion    | daili vers given author trampl snake scorpion overcom power enemi noth harm luke                     |                 |
    | 19971 | religion    | daili vers set heart eat drink worri luke                                                            |                 |
    | 19995 | religion    | daili vers whoever listen live safeti eas without fear harm proverb                                  |                 |
    | 20069 | religion    | daili vers keep perfect peac whose mind steadfast becaus trust isaiah                                |                 |
    | 20092 | religion    | daili vers dishonest money dwindl away gather money littl littl make grow proverb                    |                 |
    | 20104 | religion    | daili vers purifi yourselv obey truth sincer love brother love one anoth deepli heart ipet           |                 |
    | 20114 | religion    | daili vers command strong courag terrifi discourag lord god wherev go joshua                         |                 |
    | 20135 | religion    | immacul concept wa wa immacul forgot one thing sin fallen short glori god mark                       |                 |
    | 20144 | religion    | daili vers therefor whoever humbl like thi child greatest kingdom heaven matthew                     |                 |
    | 20153 | religion    | daili vers devot one anoth brotherli love honor one anoth abov yourselv roman                        |                 |
    | 20172 | religion    | doug sturm anyon familiar doug sturm pleas post think                                                |                 |
    | 20193 | religion    | salvat deed anoth guess salvat riddl would save joe fisher                                           |                 |
    | 20219 | religion    | prophet warn new york citi -- note repli messag                                                      |                 |
    | 20235 | religion    | propheci nyc marka traviscsdharriscom mark ashley write                                              |                 |
    | 20264 | religion    | end discuss easter close thi onc befor real tonight post                                             |                 |
    | 20293 | religion    | mission aviat fellowship hi doe anyon know anyth thi group ani info would appreci thank              |                 |
    | 20299 | religion    | daili vers overcom inherit thi hi god son revel                                                      |                 |
    | 20319 | religion    | daili vers much better get wisdom gold choos understand rather silver proverb                        |                 |
    | 20331 | religion    | daili vers whoever doe father heaven brother sister mother matthew                                   |                 |
    | 20408 | religion    | daili vers abov love deepli becaus love cover multitud sin ipet                                      |                 |
    | 20417 | religion    | love europ ani reader src go love europ congress germani thi juli -- michael davi csmcd brunelacuk   |                 |
    | 20419 | religion    | daili vers someon say faith deed show faith without deed show faith jame                             |                 |
    | 20479 | religion    | atheist view christian wa accept jeesu heart -                                                       |                 |
    | 20555 | religion    | sabbath admiss of follow thi thread talkreligion                                                     |                 |
    | 20556 | religion    | none satan                                                                                           |                 |
    | 20602 | religion    | daili vers let us becom weari good proper time reap harvest give galatian                            |                 |
    | 20612 | religion    |                                                                                                      |                 |
    | 20645 | religion    | daili vers god peac soon crush satan feet grace lord jesu roman                                      |                 |
    | 20657 | religion    | wa immacul concept note cross-post actual email thi bitlistservcathol main post goe                  |                 |
    | 20706 | religion    | translat version bibl consid accur translat                                                          |                 |
    | 20777 | religion    | daili vers peac ha made two one ha destroy barrier divid wall hostil ephesian                        |                 |
    | 20874 | religion    | daili vers receiv power holi spirit come wit jerusalem judea samaria end earth act                   |                 |
    | 20941 | religion    | daili vers given author trampl snake scorpion overcom power enemi noth harm luke                     |                 |
    | 20968 | religion    | daili vers set heart eat drink worri luke                                                            |                 |
    | 20992 | religion    | daili vers whoever listen live safeti eas without fear harm proverb                                  |                 |
    | 21066 | religion    | daili vers keep perfect peac whose mind steadfast becaus trust isaiah                                |                 |
    | 21089 | religion    | daili vers dishonest money dwindl away gather money littl littl make grow proverb                    |                 |
    | 21101 | religion    | daili vers purifi yourselv obey truth sincer love brother love one anoth deepli heart ipet           |                 |
    | 21111 | religion    | daili vers command strong courag terrifi discourag lord god wherev go joshua                         |                 |
    | 21132 | religion    | immacul concept wa wa immacul forgot one thing sin fallen short glori god mark                       |                 |
    | 21165 | religion    | apr  god promis john  hitherto ye ask noth name ask ye shall receiv joy may full john                |                 |
    | 21193 | religion    | apr  god promis john  mani receiv gave power becom son god even believ hi name john                  |                 |
    | 21287 | religion    | apr  god promis psalm  instruct thee teach thee way thou shalt go guid thee mine eye psalm           |                 |
    | 21308 | religion    | list biblic contradict critu                                                                         |                 |
    | 21381 | religion    | new religion form -- sign yawn church kibolog first better                                           |                 |
    | 21397 | religion    | on-lin book mormon newsgroup                                                                         |                 |
    | 21430 | religion    | ha risen remark heard david koresh ha risen dead dont know true thi told guy think ben l             |                 |
    | 21443 | religion    | apr  god promis proverb  tongu bring heal tree life deceit tongu crush spirit proverb  niv           |                 |
    | 21447 | religion    | info new age suggest tri expos new age dougla groothui                                               |                 |
    | 21514 | religion    | apr  god promis psalm  god whose word prais god trust afraid mortal man psalm  niv                   |                 |
    | 21536 | religion    | apr  god promis luke  said yea rather bless hear word god keep luke                                  |                 |
    | 21568 | religion    | apr  god promis psalm  look unto lighten face asham psalm                                            |                 |
    | 21618 | religion    | post say thi thread dead unfortun -- legal freedom                                                   |                 |
    | 21638 | religion    | new religion form -- sign give new yet anoth interpret odl adam eve stori -- michael                 |                 |
    | 21647 | religion    | new religion form -- sign altreligionspam                                                            |                 |
    | 21726 | religion    | apr  god promis matthew  bless hunger thirst righteous fill matthew  niv                             |                 |
    | 21734 | religion    | apr  god promis philippian  thing ye learn receiv heard seen god peac shall philippian               |                 |
    | 21793 | religion    | apr  god promis john  hitherto ye ask noth name ask ye shall receiv joy may full john                |                 |
    | 21821 | religion    | apr  god promis john  mani receiv gave power becom son god even believ hi name john                  |                 |
    | 21915 | religion    | apr  god promis psalm  instruct thee teach thee way thou shalt go guid thee mine eye psalm           |                 |
    | 21936 | religion    | list biblic contradict critu                                                                         |                 |
    | 22009 | religion    | new religion form -- sign yawn church kibolog first better                                           |                 |
    | 22025 | religion    | on-lin book mormon newsgroup                                                                         |                 |
    | 22058 | religion    | ha risen remark heard david koresh ha risen dead dont know true thi told guy think ben l             |                 |
    | 22071 | religion    | apr  god promis proverb  tongu bring heal tree life deceit tongu crush spirit proverb  niv           |                 |
    | 22075 | religion    | info new age suggest tri expos new age dougla groothui                                               |                 |
    | 22142 | religion    | apr  god promis psalm  god whose word prais god trust afraid mortal man psalm  niv                   |                 |
    | 22164 | religion    | apr  god promis luke  said yea rather bless hear word god keep luke                                  |                 |
    | 22196 | religion    | apr  god promis psalm  look unto lighten face asham psalm                                            |                 |
    | 22246 | religion    | post say thi thread dead unfortun -- legal freedom                                                   |                 |
    | 22266 | religion    | new religion form -- sign give new yet anoth interpret odl adam eve stori -- michael                 |                 |
    | 22275 | religion    | new religion form -- sign altreligionspam                                                            |                 |
    | 22354 | religion    | apr  god promis matthew  bless hunger thirst righteous fill matthew  niv                             |                 |
    | 22362 | religion    | apr  god promis philippian  thing ye learn receiv heard seen god peac shall philippian               |                 |
    | 22430 | autos       | inflat car price anyon figur pointer refer fastmuch car price gone last decad thank                  |                 |
    | 22431 | autos       | test hello test                                                                                      |                 |
    | 22440 | autos       | photo radar wa                                                                                       |                 |
    | 22460 | autos       | quick question newsgroup                                                                             |                 |
    | 22466 | autos       | awd bmw europ buy ix comput control diff rather horrid viscou coupl one outgo ix eliot               |                 |
    | 22472 | autos       | bang                                                                                                 |                 |
    | 22482 | autos       | rfd                                                                                                  |                 |
    | 22509 | autos       | bang   articl  oasysdtnavymil tobia oasysdtnavymil steve tobia write                                 |                 |
    | 22514 | autos       | brake rotor cross drill                                                                              |                 |
    | 22543 | autos       | sprayed-on bedlin info want                                                                          |                 |
    | 22545 | autos       | chang oil self                                                                                       |                 |
    | 22590 | autos       | get car need opinion good luck                                                                       |                 |
    | 22618 | autos       | mazda - doe feel right car might also need front end align particularli describ wander               |                 |
    | 22626 | autos       | manual shift bigot                                                                                   |                 |
    | 22648 | autos       | fast articl  shamanwvtekcom andrew fripwvtekcom write cocain point neither harm use care             |                 |
    | 22717 | autos       | thunderbird bought  t-bird would like ani info club around bc coast eric thoma                       |                 |
    | 22729 | autos       | no-haggl deal save sure would nt wa nt advantag mark                                                 |                 |
    | 22744 | autos       | dumbest automot concept time articl  oasysdtnavymil glouie oasysdtnavymil georg louie write          |                 |
    | 22748 | autos       | ad said nissan altima best seller guess make altima gener car us                                     |                 |
    | 22749 | autos       | no-haggl deal save local dealer advertis negoti necessari make wonder                                |                 |
    | 22764 | autos       | eagl talon tsi -- lemon                                                                              |                 |
    | 22782 | autos       | warn pleas read                                                                                      |                 |
    | 22787 | autos       | warn pleas read sorri last post server neglect send messag pleas keep thi group automot topic thank  |                 |
    | 22813 | autos       | ani honda group ani honda group especi one deal prelud tom spencer                                   |                 |
    | 22833 | autos       | impala ss go product doe mean gon na bring back biscayn bel air                                      |                 |
    | 22845 | autos       | w                                                                                                    |                 |
    | 22846 | autos       | opel owner way peopl think opel calibra                                                              |                 |
    | 22856 | autos       | opel owner -- articl ctbdg newscsouiucedu cka uxacsouiucedu oriolefan uiuc write newsgroup           |                 |
    | 22857 | autos       | opel owner -- articl apr convexcom maynard convexcom mark maynard write newsgroup                    |                 |
    | 22906 | autos       | question insur compani esp geico bob excel point correct spread word                                 |                 |
    | 22915 | autos       | lock lugnut tire rebal                                                                               |                 |
    | 22943 | autos       | shift without clutch ok take car gear without use clutch car turn thank advanc pleas repli mail eric |                 |
    | 22995 | autos       | honda mail list honda mail list subscrib                                                             |                 |
    | 23023 | autos       | v v v v vx articl apr telxonmistelxoncom joe telxonmistelxoncom joe staudt write newsgroup           |                 |
    | 23057 | autos       | ani info merc e  cheek                                                                               |                 |
    | 23061 | autos       | honda mail list excerpt netnew                                                                       |                 |
    | 23087 | autos       | bang k articl uupcb cuttinghoutxu davidbond cuttinghoutxu da vid bond write                          |                 |
    | 23138 | autos       | geico goldberg oasysdtnavymil mark goldberg write                                                    |                 |
    | 23148 | autos       | impala ss go - doe mean gon na bring back biscayn bel - air  georgehowel goucher wbffvamprorg georg  |                 |
    | 23153 | autos       | thought vw corrado vr wa curiou peopl thought vw corrado vr outatim -- -- --                         |                 |
    | 23192 | autos       | nissan nomenclatur wa manual shift bigot want                                                        |                 |
    | 23261 | autos       | v v v v vx v anyon anyon heard cizata vt mainli sold middl east dont strict legisl usa ec            |                 |
    | 23278 | autos       | servic indic bmw thank recommend decid ignor servic indic oil chang everi  mile thank respons derek  |                 |
    | 23326 | autos       | review  ford tauru sho excerpt                                                                       |                 |
    | 23420 | autos       | inflat car price anyon figur pointer refer fastmuch car price gone last decad thank                  |                 |
    | 23421 | autos       | test hello test                                                                                      |                 |
    | 23430 | autos       | photo radar wa                                                                                       |                 |
    | 23449 | autos       | quick question newsgroup                                                                             |                 |
    | 23455 | autos       | awd bmw europ buy ix comput control diff rather horrid viscou coupl one outgo ix eliot               |                 |
    | 23461 | autos       | bang                                                                                                 |                 |
    | 23465 | autos       | photo radar wa                                                                                       |                 |
    | 23472 | autos       | rfd                                                                                                  |                 |
    | 23499 | autos       | bang   articl  oasysdtnavymil tobia oasysdtnavymil steve tobia write                                 |                 |
    | 23504 | autos       | brake rotor cross drill                                                                              |                 |
    | 23533 | autos       | sprayed-on bedlin info want                                                                          |                 |
    | 23535 | autos       | chang oil self                                                                                       |                 |
    | 23541 | autos       |                                                                                                      |                 |
    | 23542 | autos       |                                                                                                      |                 |
    | 23546 | autos       |                                                                                                      |                 |
    | 23547 | autos       |                                                                                                      |                 |
    | 23548 | autos       |                                                                                                      |                 |
    | 23549 | autos       |                                                                                                      |                 |
    | 23580 | autos       | get car need opinion good luck                                                                       |                 |
    | 23608 | autos       | mazda - doe feel right car might also need front end align particularli describ wander               |                 |
    | 23616 | autos       | manual shift bigot                                                                                   |                 |
    | 23638 | autos       | fast articl  shamanwvtekcom andrew fripwvtekcom write cocain point neither harm use care             |                 |
    | 23707 | autos       | thunderbird bought  t-bird would like ani info club around bc coast eric thoma                       |                 |
    | 23719 | autos       | no-haggl deal save sure would nt wa nt advantag mark                                                 |                 |
    | 23734 | autos       | dumbest automot concept time articl  oasysdtnavymil glouie oasysdtnavymil georg louie write          |                 |
    | 23738 | autos       | ad said nissan altima best seller guess make altima gener car us                                     |                 |
    | 23739 | autos       | no-haggl deal save local dealer advertis negoti necessari make wonder                                |                 |
    | 23754 | autos       | eagl talon tsi -- lemon                                                                              |                 |
    | 23772 | autos       | warn pleas read                                                                                      |                 |
    | 23803 | autos       | ani honda group ani honda group especi one deal prelud tom spencer                                   |                 |
    | 23823 | autos       | impala ss go product doe mean gon na bring back biscayn bel air                                      |                 |
    | 23835 | autos       | w                                                                                                    |                 |
    | 23836 | autos       | opel owner way peopl think opel calibra                                                              |                 |
    | 23846 | autos       | opel owner -- articl ctbdg newscsouiucedu cka uxacsouiucedu oriolefan uiuc write newsgroup           |                 |
    | 23847 | autos       | opel owner -- articl apr convexcom maynard convexcom mark maynard write newsgroup                    |                 |
    | 23896 | autos       | question insur compani esp geico bob excel point correct spread word                                 |                 |
    | 23905 | autos       | lock lugnut tire rebal                                                                               |                 |
    | 23985 | autos       | honda mail list honda mail list subscrib                                                             |                 |
    | 24013 | autos       | v v v v vx articl apr telxonmistelxoncom joe telxonmistelxoncom joe staudt write newsgroup           |                 |
    | 24047 | autos       | ani info merc e  cheek                                                                               |                 |
    | 24051 | autos       | honda mail list excerpt netnew                                                                       |                 |
    | 24077 | autos       | bang k articl uupcb cuttinghoutxu davidbond cuttinghoutxu da vid bond write                          |                 |
    | 24128 | autos       | geico goldberg oasysdtnavymil mark goldberg write                                                    |                 |
    | 24143 | autos       | thought vw corrado vr wa curiou peopl thought vw corrado vr outatim -- -- --                         |                 |
    | 24182 | autos       | nissan nomenclatur wa manual shift bigot want                                                        |                 |
    | 24251 | autos       | v v v v vx v anyon anyon heard cizata vt mainli sold middl east dont strict legisl usa ec            |                 |
    | 24316 | autos       | review  ford tauru sho excerpt                                                                       |                 |
    | 24390 | autos       | pink tool wa girlfriend motorcycl onli prevent dive thi one                                          |                 |
    | 24409 | autos       | rm split wa insect impact hpcc                                                                       |                 |
    | 24413 | autos       | tool tool tool flank drive everyon talk                                                              |                 |
    | 24442 | autos       | goldw perform hpcc                                                                                   |                 |
    | 24452 | autos       | mgnoc address anyon ha current moto guzzi nation owner club address pleas e-mail thank advanc tk --  |                 |
    | 24491 | autos       | brake rotor cross drill ---                                                                          |                 |
    | 24505 | autos       | april  wa faq - dod articl  hpfcsofchpcom jld hpfcsofchpcom jeff deeney write                        |                 |
    | 24511 | autos       | v-max handl request hello ican anyon ha handson experi ride yamaha v-max pl kindli comment handl     |                 |
    | 24546 | autos       | dot tire date code                                                                                   |                 |
    | 24555 | autos       | rejet carb jump middl thi thread may know yall talk comment                                          |                 |
    | 24576 | autos       | kawasaki  ae sale includ tha sale cover cover sold separ trailer sold pat                            |                 |
    | 24584 | autos       | riceburn respect                                                                                     |                 |
    | 24604 | autos       | live free quietli die                                                                                |                 |
    | 24667 | autos       | bmw moa member read thi resign bmw moa get remaind -year membership refund                           |                 |
    | 24714 | autos       | fortune-guzzl bar bar okay perfectli welcom come scotland know -                                     |                 |
    | 24740 | autos       | honda cbf sale want let peopl know thi motorcycl ha sold thank inquiri -- dave schultz               |                 |
    | 24756 | autos       | help bike short                                                                                      |                 |
    | 24758 | autos       | shaft-driv wheeli possibl wheeli motorcycl shaft-driv mike terri  virago                             |                 |
    | 24860 | autos       | invert fork need                                                                                     |                 |
    | 24873 | autos       | help bike short sure older bike yamaha virago  ha spec seat height  honda shadow                     |                 |
    | 24933 | autos       | protect gear second boot oil spot car particularli slipperi park bike good boot help well -- squid   |                 |
    | 24946 | autos       | xs time could kind soul tell advanc timingrev  xs special bought canada thank                        |                 |
    | 24993 | autos       | use bike east vs west coast hpcc                                                                     |                 |
    | 25011 | autos       | want advic new cylist angel levin write exactli danger look anyon particular mind jodi -             |                 |
    | 25043 | autos       | dog articl apr acsucalgaryca parr acsucalgaryca charl parr write newsgroup                           |                 |
    | 25078 | autos       | type spesif cb vfr gt etc let forget st sport tour honda                                             |                 |
    | 25108 | autos       | moa member anoth letter read ha anyon notic thi happen sinc chri perez wa gift membership anyon      |                 |
    | 25118 | autos       | ok wa littl hasti                                                                                    |                 |
    | 25119 | autos       | well blow yuk yuk yuk                                                                                |                 |
    | 25137 | autos       | test test                                                                                            |                 |
    | 25203 | autos       | look movi w bike greas ii arun cool rider -- noe look eye like black hole sky shine crazi diamond    |                 |
    | 25235 | autos       | type spesif cb vfr gt etc                                                                            |                 |
    | 25236 | autos       | first bike wheeli                                                                                    |                 |
    | 25241 | autos       | test ignor ignor                                                                                     |                 |
    | 25271 | autos       | first bike                                                                                           |                 |
    | 25285 | autos       | help adjust tappit                                                                                   |                 |
    | 25340 | autos       | shaft-driv wheeli  hpcccorphpcom gharriso hpcccorphpcom graem harrison write hpcc                    |                 |
    | 25341 | autos       | type spesif cb vfr gt etc victor johnson thu  apr   gmt wibbl                                        |                 |
    | 25384 | autos       | pink tool wa girlfriend motorcycl onli prevent dive thi one                                          |                 |
    | 25403 | autos       | rm split wa insect impact hpcc                                                                       |                 |
    | 25407 | autos       | tool tool tool flank drive everyon talk                                                              |                 |
    | 25436 | autos       | goldw perform hpcc                                                                                   |                 |
    | 25485 | autos       | brake rotor cross drill ---                                                                          |                 |
    | 25499 | autos       | april  wa faq - dod articl  hpfcsofchpcom jld hpfcsofchpcom jeff deeney write                        |                 |
    | 25505 | autos       | v-max handl request hello ican anyon ha handson experi ride yamaha v-max pl kindli comment handl     |                 |
    | 25540 | autos       | dot tire date code                                                                                   |                 |
    | 25549 | autos       | rejet carb jump middl thi thread may know yall talk comment                                          |                 |
    | 25570 | autos       | kawasaki  ae sale includ tha sale cover cover sold separ trailer sold pat                            |                 |
    | 25578 | autos       | riceburn respect                                                                                     |                 |
    | 25598 | autos       | live free quietli die                                                                                |                 |
    | 25661 | autos       | bmw moa member read thi resign bmw moa get remaind -year membership refund                           |                 |
    | 25708 | autos       | fortune-guzzl bar bar okay perfectli welcom come scotland know -                                     |                 |
    | 25734 | autos       | honda cbf sale want let peopl know thi motorcycl ha sold thank inquiri -- dave schultz               |                 |
    | 25750 | autos       | help bike short                                                                                      |                 |
    | 25752 | autos       | shaft-driv wheeli possibl wheeli motorcycl shaft-driv mike terri  virago                             |                 |
    | 25854 | autos       | invert fork need                                                                                     |                 |
    | 25867 | autos       | help bike short sure older bike yamaha virago  ha spec seat height  honda shadow                     |                 |
    | 25927 | autos       | protect gear second boot oil spot car particularli slipperi park bike good boot help well -- squid   |                 |
    | 25940 | autos       | xs time could kind soul tell advanc timingrev  xs special bought canada thank                        |                 |
    | 25987 | autos       | use bike east vs west coast hpcc                                                                     |                 |
    | 26005 | autos       | want advic new cylist angel levin write exactli danger look anyon particular mind jodi -             |                 |
    | 26037 | autos       | dog articl apr acsucalgaryca parr acsucalgaryca charl parr write newsgroup                           |                 |
    | 26072 | autos       | type spesif cb vfr gt etc let forget st sport tour honda                                             |                 |
    | 26102 | autos       | moa member anoth letter read ha anyon notic thi happen sinc chri perez wa gift membership anyon      |                 |
    | 26112 | autos       | ok wa littl hasti                                                                                    |                 |
    | 26113 | autos       | well blow yuk yuk yuk                                                                                |                 |
    | 26131 | autos       | test test                                                                                            |                 |
    | 26197 | autos       | look movi w bike greas ii arun cool rider -- noe look eye like black hole sky shine crazi diamond    |                 |
    | 26229 | autos       | type spesif cb vfr gt etc                                                                            |                 |
    | 26230 | autos       | first bike wheeli                                                                                    |                 |
    | 26235 | autos       | test ignor ignor                                                                                     |                 |
    | 26265 | autos       | first bike                                                                                           |                 |
    | 26279 | autos       | help adjust tappit                                                                                   |                 |
    | 26334 | autos       | shaft-driv wheeli  hpcccorphpcom gharriso hpcccorphpcom graem harrison write hpcc                    |                 |
    | 26335 | autos       | type spesif cb vfr gt etc victor johnson thu  apr   gmt wibbl                                        |                 |
    | 26403 | comp_elec   | graphic librari packag                                                                               |                 |
    | 26439 | comp_elec   | test sorri                                                                                           |                 |
    | 26534 | comp_elec   | nice gif code thing call xgif xgif grandfath xv --  --                                               |                 |
    | 26649 | comp_elec   | virtual realiti x cheap updat locat directori publicvirtual-world sorri - robert robert acsccom      |                 |
    | 26763 | comp_elec   | pcx articl apr freenetcarletonca ad freenetcarletonca jason wiggl write newsgroup                    |                 |
    | 26820 | comp_elec   | pleas recommend d graphic librari f sorri mention platform origin post wa macprogramm decid post     |                 |
    | 26898 | comp_elec   | sphere  point newsgroup                                                                              |                 |
    | 27171 | comp_elec   | vesa speedstar  post john cormack want tell slight differ speedstar  speedstar x stefan              |                 |
    | 28154 | comp_elec   | need viewer gl file hi subject say pd viewer gl file x thank dominik                                 |                 |
    | 28157 | comp_elec   | radios radios sourc want read                                                                        |                 |
    | 28163 | comp_elec   | radios articl  rzuni-jenad hahm fossihab-weimard peter hahm write radios sourc want read             |                 |
    | 28315 | comp_elec   | m-motion video card yuv rgb contact offlin thi rick                                                  |                 |
    | 28463 | comp_elec   | coreldraw bitmap scodal coreldraw whatev write scodl file directli look fileexport main menu rick    |                 |
    | 28585 | comp_elec   | newsgroup split nt follow thi thread appolog thi ha alreadi mention                                  |                 |
    | 28739 | comp_elec   | hpxx precompil version hpxx - prefer                                                                 |                 |
    | 28767 | comp_elec   | grayscal printer consid appl laserwrit iig use b w imag print                                        |                 |
    | 28810 | comp_elec   | postscript draw prog articl apr informatiktu-muenchend                                               |                 |
    | 28843 | comp_elec   | gif aerial map ftp site map us prefer aerial photograph thank                                        |                 |
    | 28864 | comp_elec   | technic help sought                                                                                  |                 |
    | 28867 | comp_elec   | autocad - tiff done                                                                                  |                 |
    | 28870 | comp_elec   | e-mail michael abrash                                                                                |                 |
    | 28873 | comp_elec   | xga- info                                                                                            |                 |
    | 28876 | comp_elec   | render softwar multi-processor comput                                                                |                 |
    | 28879 | comp_elec   | announc ivan sutherland speak harvard                                                                |                 |
    | 28882 | comp_elec   | newss                                                                                                |                 |
    | 28885 | comp_elec   | pd d viewer want                                                                                     |                 |
    | 28888 | comp_elec   | gl fli spec                                                                                          |                 |
    | 28891 | comp_elec   | video inout                                                                                          |                 |
    | 28894 | comp_elec   | xv ms-do                                                                                             |                 |
    | 28897 | comp_elec   | need rgb data save imag                                                                              |                 |
    | 28900 | comp_elec   | univesa driver                                                                                       |                 |
    | 28903 | comp_elec   | xv ms-do                                                                                             |                 |
    | 28906 | comp_elec   | tri view pov file                                                                                    |                 |
    | 28909 | comp_elec   | cornerston dualpag driver want                                                                       |                 |
    | 28912 | comp_elec   | march cub                                                                                            |                 |
    | 28915 | comp_elec   | xlib  bit display info need                                                                          |                 |
    | 28918 | comp_elec   | pov file constructor unixx                                                                           |                 |
    | 28921 | comp_elec   | look tseng vesa driver                                                                               |                 |
    | 28962 | comp_elec   | packag fashion design thi articl wa probabl gener buggi news reader                                  |                 |
    | 28972 | comp_elec   | attract draw sphere subscrib                                                                         |                 |
    | 28998 | comp_elec   | look usa map doe anyon know line draw usa map thank veri much advanc hoi yoo engrucfedu              |                 |
    | 29078 | comp_elec   | ftp achiv usg terrain dat tri spectrumxeroxcom  pubmapdem                                            |                 |
    | 29084 | comp_elec   | apr                                                                                                  |                 |
    | 29128 | comp_elec   | vrrend kept told vrrend avail internet want know thank advanc raoul daruwala csnyuedu                |                 |
    | 29140 | comp_elec   | front end povray                                                                                     |                 |
    | 29191 | comp_elec   | faq find thank kwansik                                                                               |                 |
    | 29197 | comp_elec   | dxf pcx gif tif tga                                                                                  |                 |
    | 29202 | comp_elec   | help d studio ipa hi anyon pleas give ftp site get ipa process d studio  thank warren                |                 |
    | 29206 | comp_elec   | dwggcddd format refer need kind soul point refer abov format thank earl                              |                 |
    | 29214 | comp_elec   | dna helix                                                                                            |                 |
    | 29215 | comp_elec   | look tiffep dna helix e-mail ani auggest pleas                                                       |                 |
    | 29247 | comp_elec   | povray tga - rle                                                                                     |                 |
    | 29256 | comp_elec   | virtual realiti x cheap                                                                              |                 |
    | 29259 | comp_elec   | look graig toontown notic post                                                                       |                 |
    | 29271 | comp_elec   | sourc codehelp ip packag pleas                                                                       |                 |
    | 29294 | comp_elec   | geospher imag articl  altgraph newsgroup altgraph path newsndedu molier rmalayt                      |                 |
    | 29308 | comp_elec   | sw convert plot ascii file look softwar read plot pcx format convert x coordin                       |                 |
    | 29346 | comp_elec   | gif targa hello subject say need gif targa convert dta could make fli krzysztof                      |                 |
    | 29392 | comp_elec   | phig user group confer                                                                               |                 |
    | 29433 | comp_elec   | graphic design newsgroup newsgroup discuss graphic design pc mac yknow like corel draw               |                 |
    | 29459 | comp_elec   | pbm dec whensthenewvers                                                                              |                 |
    | 29460 | comp_elec   | doe anyon know fabl new version pbm soon far know current version dec jeff p jeffrey e hundstad      |                 |
    | 29484 | comp_elec   | gw anybodi know get graphic work shop brad utkvxutkedu                                               |                 |
    | 29496 | comp_elec   | tiff complex                                                                                         |                 |
    | 29521 | comp_elec   | art letter graphic editor                                                                            |                 |
    | 29553 | comp_elec   | do someon pleas fill do thank bh                                                                     |                 |
    | 29573 | comp_elec   | gif targa                                                                                            |                 |
    | 29606 | comp_elec   | adob photo shop type softwar unixxmotif platform articl cwxbiv worldstdcom sciimageprocess           |                 |
    | 29610 | comp_elec   | fractal terrain gener                                                                                |                 |
    | 29621 | comp_elec   | d graphic softwar compani info believ mani peopl happi thi inform pleas post                         |                 |
    | 29723 | comp_elec   | gamma correct someon know talk add faq entri gamma correct thank mark                                |                 |
    | 29729 | comp_elec   | autodesk bb doe autodesk ha bb --                                                                    |                 |
    | 29731 | comp_elec   | siggraph onlin experiment public avail tri cding publicationsmayonlin siggraphorg rosale             |                 |
    | 29769 | comp_elec   | calcul regular polyhedra vertic interest copi thi code run across mail author bounc hpldsla          |                 |
    | 29845 | comp_elec   |                                                                                                      |                 |
    | 29846 | comp_elec   |                                                                                                      |                 |
    | 29847 | comp_elec   |                                                                                                      |                 |
    | 29848 | comp_elec   |                                                                                                      |                 |
    | 29849 | comp_elec   |                                                                                                      |                 |
    | 29850 | comp_elec   |                                                                                                      |                 |
    | 29851 | comp_elec   |                                                                                                      |                 |
    | 29852 | comp_elec   |                                                                                                      |                 |
    | 29853 | comp_elec   |                                                                                                      |                 |
    | 29854 | comp_elec   |                                                                                                      |                 |
    | 29855 | comp_elec   |                                                                                                      |                 |
    | 29856 | comp_elec   |                                                                                                      |                 |
    | 29857 | comp_elec   |                                                                                                      |                 |
    | 29858 | comp_elec   |                                                                                                      |                 |
    | 29859 | comp_elec   |                                                                                                      |                 |
    | 29860 | comp_elec   |                                                                                                      |                 |
    | 29861 | comp_elec   |                                                                                                      |                 |
    | 29862 | comp_elec   |                                                                                                      |                 |
    | 29863 | comp_elec   |                                                                                                      |                 |
    | 29864 | comp_elec   |                                                                                                      |                 |
    | 29865 | comp_elec   |                                                                                                      |                 |
    | 29866 | comp_elec   |                                                                                                      |                 |
    | 29867 | comp_elec   |                                                                                                      |                 |
    | 29868 | comp_elec   |                                                                                                      |                 |
    | 29869 | comp_elec   |                                                                                                      |                 |
    | 29870 | comp_elec   |                                                                                                      |                 |
    | 29871 | comp_elec   |                                                                                                      |                 |
    | 29872 | comp_elec   |                                                                                                      |                 |
    | 29873 | comp_elec   | technic help sought                                                                                  |                 |
    | 29874 | comp_elec   |                                                                                                      |                 |
    | 29875 | comp_elec   |                                                                                                      |                 |
    | 29876 | comp_elec   |                                                                                                      |                 |
    | 29877 | comp_elec   |                                                                                                      |                 |
    | 29878 | comp_elec   |                                                                                                      |                 |
    | 29879 | comp_elec   |                                                                                                      |                 |
    | 29880 | comp_elec   |                                                                                                      |                 |
    | 29881 | comp_elec   |                                                                                                      |                 |
    | 29882 | comp_elec   |                                                                                                      |                 |
    | 29883 | comp_elec   |                                                                                                      |                 |
    | 29884 | comp_elec   |                                                                                                      |                 |
    | 29885 | comp_elec   |                                                                                                      |                 |
    | 29886 | comp_elec   |                                                                                                      |                 |
    | 29887 | comp_elec   |                                                                                                      |                 |
    | 29888 | comp_elec   |                                                                                                      |                 |
    | 29889 | comp_elec   |                                                                                                      |                 |
    | 29890 | comp_elec   |                                                                                                      |                 |
    | 29891 | comp_elec   |                                                                                                      |                 |
    | 29892 | comp_elec   |                                                                                                      |                 |
    | 29893 | comp_elec   |                                                                                                      |                 |
    | 29894 | comp_elec   | autocad - tiff done                                                                                  |                 |
    | 29895 | comp_elec   |                                                                                                      |                 |
    | 29896 | comp_elec   |                                                                                                      |                 |
    | 29897 | comp_elec   |                                                                                                      |                 |
    | 29898 | comp_elec   |                                                                                                      |                 |
    | 29899 | comp_elec   |                                                                                                      |                 |
    | 29900 | comp_elec   |                                                                                                      |                 |
    | 29901 | comp_elec   |                                                                                                      |                 |
    | 29902 | comp_elec   |                                                                                                      |                 |
    | 29903 | comp_elec   |                                                                                                      |                 |
    | 29904 | comp_elec   |                                                                                                      |                 |
    | 29905 | comp_elec   |                                                                                                      |                 |
    | 29906 | comp_elec   |                                                                                                      |                 |
    | 29907 | comp_elec   |                                                                                                      |                 |
    | 29908 | comp_elec   |                                                                                                      |                 |
    | 29909 | comp_elec   |                                                                                                      |                 |
    | 29910 | comp_elec   |                                                                                                      |                 |
    | 29911 | comp_elec   |                                                                                                      |                 |
    | 29912 | comp_elec   |                                                                                                      |                 |
    | 29913 | comp_elec   |                                                                                                      |                 |
    | 29914 | comp_elec   |                                                                                                      |                 |
    | 29915 | comp_elec   |                                                                                                      |                 |
    | 29916 | comp_elec   | e-mail michael abrash                                                                                |                 |
    | 29917 | comp_elec   |                                                                                                      |                 |
    | 29918 | comp_elec   |                                                                                                      |                 |
    | 29919 | comp_elec   |                                                                                                      |                 |
    | 29920 | comp_elec   |                                                                                                      |                 |
    | 29921 | comp_elec   |                                                                                                      |                 |
    | 29922 | comp_elec   |                                                                                                      |                 |
    | 29923 | comp_elec   |                                                                                                      |                 |
    | 29924 | comp_elec   |                                                                                                      |                 |
    | 29925 | comp_elec   |                                                                                                      |                 |
    | 29926 | comp_elec   |                                                                                                      |                 |
    | 29927 | comp_elec   |                                                                                                      |                 |
    | 29928 | comp_elec   |                                                                                                      |                 |
    | 29929 | comp_elec   |                                                                                                      |                 |
    | 29930 | comp_elec   |                                                                                                      |                 |
    | 29931 | comp_elec   |                                                                                                      |                 |
    | 29932 | comp_elec   |                                                                                                      |                 |
    | 29933 | comp_elec   |                                                                                                      |                 |
    | 29934 | comp_elec   |                                                                                                      |                 |
    | 29935 | comp_elec   |                                                                                                      |                 |
    | 29936 | comp_elec   |                                                                                                      |                 |
    | 29937 | comp_elec   |                                                                                                      |                 |
    | 29938 | comp_elec   |                                                                                                      |                 |
    | 29939 | comp_elec   |                                                                                                      |                 |
    | 29940 | comp_elec   |                                                                                                      |                 |
    | 29941 | comp_elec   |                                                                                                      |                 |
    | 29942 | comp_elec   |                                                                                                      |                 |
    | 29943 | comp_elec   |                                                                                                      |                 |
    | 29944 | comp_elec   |                                                                                                      |                 |
    | 29945 | comp_elec   |                                                                                                      |                 |
    | 29946 | comp_elec   |                                                                                                      |                 |
    | 29947 | comp_elec   |                                                                                                      |                 |
    | 29948 | comp_elec   |                                                                                                      |                 |
    | 29949 | comp_elec   |                                                                                                      |                 |
    | 29950 | comp_elec   |                                                                                                      |                 |
    | 29951 | comp_elec   |                                                                                                      |                 |
    | 29952 | comp_elec   |                                                                                                      |                 |
    | 29953 | comp_elec   |                                                                                                      |                 |
    | 29954 | comp_elec   |                                                                                                      |                 |
    | 29955 | comp_elec   |                                                                                                      |                 |
    | 29956 | comp_elec   |                                                                                                      |                 |
    | 29957 | comp_elec   |                                                                                                      |                 |
    | 29958 | comp_elec   |                                                                                                      |                 |
    | 29959 | comp_elec   |                                                                                                      |                 |
    | 29960 | comp_elec   |                                                                                                      |                 |
    | 29961 | comp_elec   |                                                                                                      |                 |
    | 29962 | comp_elec   |                                                                                                      |                 |
    | 29963 | comp_elec   |                                                                                                      |                 |
    | 29964 | comp_elec   |                                                                                                      |                 |
    | 29965 | comp_elec   |                                                                                                      |                 |
    | 29966 | comp_elec   |                                                                                                      |                 |
    | 29967 | comp_elec   |                                                                                                      |                 |
    | 29968 | comp_elec   |                                                                                                      |                 |
    | 29969 | comp_elec   |                                                                                                      |                 |
    | 29970 | comp_elec   |                                                                                                      |                 |
    | 29971 | comp_elec   |                                                                                                      |                 |
    | 29972 | comp_elec   |                                                                                                      |                 |
    | 29973 | comp_elec   |                                                                                                      |                 |
    | 29974 | comp_elec   |                                                                                                      |                 |
    | 29975 | comp_elec   |                                                                                                      |                 |
    | 29976 | comp_elec   |                                                                                                      |                 |
    | 29977 | comp_elec   |                                                                                                      |                 |
    | 29978 | comp_elec   |                                                                                                      |                 |
    | 29979 | comp_elec   |                                                                                                      |                 |
    | 29980 | comp_elec   |                                                                                                      |                 |
    | 29981 | comp_elec   |                                                                                                      |                 |
    | 29982 | comp_elec   | xga- info                                                                                            |                 |
    | 29983 | comp_elec   |                                                                                                      |                 |
    | 29984 | comp_elec   |                                                                                                      |                 |
    | 29985 | comp_elec   | render softwar multi-processor comput                                                                |                 |
    | 29986 | comp_elec   |                                                                                                      |                 |
    | 29987 | comp_elec   |                                                                                                      |                 |
    | 29988 | comp_elec   |                                                                                                      |                 |
    | 29989 | comp_elec   |                                                                                                      |                 |
    | 29990 | comp_elec   |                                                                                                      |                 |
    | 29991 | comp_elec   |                                                                                                      |                 |
    | 29992 | comp_elec   |                                                                                                      |                 |
    | 29993 | comp_elec   |                                                                                                      |                 |
    | 29994 | comp_elec   | announc ivan sutherland speak harvard                                                                |                 |
    | 29995 | comp_elec   |                                                                                                      |                 |
    | 29996 | comp_elec   |                                                                                                      |                 |
    | 29997 | comp_elec   |                                                                                                      |                 |
    | 29998 | comp_elec   |                                                                                                      |                 |
    | 29999 | comp_elec   |                                                                                                      |                 |
    | 30000 | comp_elec   |                                                                                                      |                 |
    | 30001 | comp_elec   |                                                                                                      |                 |
    | 30002 | comp_elec   |                                                                                                      |                 |
    | 30003 | comp_elec   |                                                                                                      |                 |
    | 30004 | comp_elec   |                                                                                                      |                 |
    | 30005 | comp_elec   |                                                                                                      |                 |
    | 30006 | comp_elec   |                                                                                                      |                 |
    | 30007 | comp_elec   |                                                                                                      |                 |
    | 30008 | comp_elec   |                                                                                                      |                 |
    | 30009 | comp_elec   |                                                                                                      |                 |
    | 30010 | comp_elec   |                                                                                                      |                 |
    | 30011 | comp_elec   |                                                                                                      |                 |
    | 30012 | comp_elec   |                                                                                                      |                 |
    | 30013 | comp_elec   |                                                                                                      |                 |
    | 30014 | comp_elec   |                                                                                                      |                 |
    | 30015 | comp_elec   |                                                                                                      |                 |
    | 30016 | comp_elec   |                                                                                                      |                 |
    | 30017 | comp_elec   |                                                                                                      |                 |
    | 30018 | comp_elec   |                                                                                                      |                 |
    | 30019 | comp_elec   |                                                                                                      |                 |
    | 30020 | comp_elec   |                                                                                                      |                 |
    | 30021 | comp_elec   |                                                                                                      |                 |
    | 30022 | comp_elec   |                                                                                                      |                 |
    | 30023 | comp_elec   |                                                                                                      |                 |
    | 30024 | comp_elec   |                                                                                                      |                 |
    | 30025 | comp_elec   |                                                                                                      |                 |
    | 30026 | comp_elec   |                                                                                                      |                 |
    | 30027 | comp_elec   |                                                                                                      |                 |
    | 30028 | comp_elec   |                                                                                                      |                 |
    | 30029 | comp_elec   |                                                                                                      |                 |
    | 30030 | comp_elec   |                                                                                                      |                 |
    | 30031 | comp_elec   |                                                                                                      |                 |
    | 30032 | comp_elec   |                                                                                                      |                 |
    | 30033 | comp_elec   |                                                                                                      |                 |
    | 30034 | comp_elec   |                                                                                                      |                 |
    | 30035 | comp_elec   |                                                                                                      |                 |
    | 30036 | comp_elec   |                                                                                                      |                 |
    | 30037 | comp_elec   |                                                                                                      |                 |
    | 30038 | comp_elec   |                                                                                                      |                 |
    | 30039 | comp_elec   |                                                                                                      |                 |
    | 30040 | comp_elec   |                                                                                                      |                 |
    | 30041 | comp_elec   |                                                                                                      |                 |
    | 30042 | comp_elec   |                                                                                                      |                 |
    | 30043 | comp_elec   |                                                                                                      |                 |
    | 30044 | comp_elec   |                                                                                                      |                 |
    | 30045 | comp_elec   |                                                                                                      |                 |
    | 30046 | comp_elec   |                                                                                                      |                 |
    | 30047 | comp_elec   |                                                                                                      |                 |
    | 30048 | comp_elec   |                                                                                                      |                 |
    | 30049 | comp_elec   |                                                                                                      |                 |
    | 30050 | comp_elec   |                                                                                                      |                 |
    | 30051 | comp_elec   |                                                                                                      |                 |
    | 30052 | comp_elec   |                                                                                                      |                 |
    | 30053 | comp_elec   |                                                                                                      |                 |
    | 30054 | comp_elec   |                                                                                                      |                 |
    | 30055 | comp_elec   | newss                                                                                                |                 |
    | 30056 | comp_elec   | pd d viewer want                                                                                     |                 |
    | 30057 | comp_elec   |                                                                                                      |                 |
    | 30058 | comp_elec   |                                                                                                      |                 |
    | 30059 | comp_elec   |                                                                                                      |                 |
    | 30060 | comp_elec   |                                                                                                      |                 |
    | 30061 | comp_elec   |                                                                                                      |                 |
    | 30062 | comp_elec   |                                                                                                      |                 |
    | 30063 | comp_elec   |                                                                                                      |                 |
    | 30064 | comp_elec   |                                                                                                      |                 |
    | 30065 | comp_elec   |                                                                                                      |                 |
    | 30066 | comp_elec   |                                                                                                      |                 |
    | 30067 | comp_elec   |                                                                                                      |                 |
    | 30068 | comp_elec   |                                                                                                      |                 |
    | 30069 | comp_elec   |                                                                                                      |                 |
    | 30070 | comp_elec   |                                                                                                      |                 |
    | 30071 | comp_elec   |                                                                                                      |                 |
    | 30072 | comp_elec   |                                                                                                      |                 |
    | 30073 | comp_elec   |                                                                                                      |                 |
    | 30074 | comp_elec   |                                                                                                      |                 |
    | 30075 | comp_elec   |                                                                                                      |                 |
    | 30076 | comp_elec   |                                                                                                      |                 |
    | 30077 | comp_elec   |                                                                                                      |                 |
    | 30078 | comp_elec   |                                                                                                      |                 |
    | 30079 | comp_elec   |                                                                                                      |                 |
    | 30080 | comp_elec   |                                                                                                      |                 |
    | 30081 | comp_elec   |                                                                                                      |                 |
    | 30082 | comp_elec   | gl fli spec                                                                                          |                 |
    | 30083 | comp_elec   |                                                                                                      |                 |
    | 30084 | comp_elec   |                                                                                                      |                 |
    | 30085 | comp_elec   |                                                                                                      |                 |
    | 30086 | comp_elec   |                                                                                                      |                 |
    | 30087 | comp_elec   |                                                                                                      |                 |
    | 30088 | comp_elec   |                                                                                                      |                 |
    | 30089 | comp_elec   |                                                                                                      |                 |
    | 30090 | comp_elec   |                                                                                                      |                 |
    | 30091 | comp_elec   |                                                                                                      |                 |
    | 30092 | comp_elec   |                                                                                                      |                 |
    | 30093 | comp_elec   |                                                                                                      |                 |
    | 30094 | comp_elec   |                                                                                                      |                 |
    | 30095 | comp_elec   |                                                                                                      |                 |
    | 30096 | comp_elec   |                                                                                                      |                 |
    | 30097 | comp_elec   |                                                                                                      |                 |
    | 30098 | comp_elec   |                                                                                                      |                 |
    | 30099 | comp_elec   |                                                                                                      |                 |
    | 30100 | comp_elec   |                                                                                                      |                 |
    | 30101 | comp_elec   |                                                                                                      |                 |
    | 30102 | comp_elec   |                                                                                                      |                 |
    | 30103 | comp_elec   |                                                                                                      |                 |
    | 30104 | comp_elec   |                                                                                                      |                 |
    | 30105 | comp_elec   |                                                                                                      |                 |
    | 30106 | comp_elec   |                                                                                                      |                 |
    | 30107 | comp_elec   |                                                                                                      |                 |
    | 30108 | comp_elec   |                                                                                                      |                 |
    | 30109 | comp_elec   |                                                                                                      |                 |
    | 30110 | comp_elec   |                                                                                                      |                 |
    | 30111 | comp_elec   |                                                                                                      |                 |
    | 30112 | comp_elec   |                                                                                                      |                 |
    | 30113 | comp_elec   |                                                                                                      |                 |
    | 30114 | comp_elec   |                                                                                                      |                 |
    | 30115 | comp_elec   |                                                                                                      |                 |
    | 30116 | comp_elec   |                                                                                                      |                 |
    | 30117 | comp_elec   |                                                                                                      |                 |
    | 30118 | comp_elec   |                                                                                                      |                 |
    | 30119 | comp_elec   |                                                                                                      |                 |
    | 30120 | comp_elec   |                                                                                                      |                 |
    | 30121 | comp_elec   |                                                                                                      |                 |
    | 30122 | comp_elec   |                                                                                                      |                 |
    | 30123 | comp_elec   |                                                                                                      |                 |
    | 30124 | comp_elec   |                                                                                                      |                 |
    | 30125 | comp_elec   |                                                                                                      |                 |
    | 30126 | comp_elec   |                                                                                                      |                 |
    | 30127 | comp_elec   |                                                                                                      |                 |
    | 30128 | comp_elec   |                                                                                                      |                 |
    | 30129 | comp_elec   |                                                                                                      |                 |
    | 30130 | comp_elec   |                                                                                                      |                 |
    | 30131 | comp_elec   |                                                                                                      |                 |
    | 30132 | comp_elec   |                                                                                                      |                 |
    | 30133 | comp_elec   |                                                                                                      |                 |
    | 30134 | comp_elec   |                                                                                                      |                 |
    | 30135 | comp_elec   |                                                                                                      |                 |
    | 30136 | comp_elec   |                                                                                                      |                 |
    | 30137 | comp_elec   |                                                                                                      |                 |
    | 30138 | comp_elec   |                                                                                                      |                 |
    | 30139 | comp_elec   |                                                                                                      |                 |
    | 30140 | comp_elec   |                                                                                                      |                 |
    | 30141 | comp_elec   |                                                                                                      |                 |
    | 30142 | comp_elec   |                                                                                                      |                 |
    | 30143 | comp_elec   |                                                                                                      |                 |
    | 30144 | comp_elec   |                                                                                                      |                 |
    | 30145 | comp_elec   |                                                                                                      |                 |
    | 30146 | comp_elec   |                                                                                                      |                 |
    | 30147 | comp_elec   |                                                                                                      |                 |
    | 30148 | comp_elec   |                                                                                                      |                 |
    | 30149 | comp_elec   |                                                                                                      |                 |
    | 30150 | comp_elec   |                                                                                                      |                 |
    | 30151 | comp_elec   |                                                                                                      |                 |
    | 30152 | comp_elec   | video inout                                                                                          |                 |
    | 30153 | comp_elec   |                                                                                                      |                 |
    | 30154 | comp_elec   |                                                                                                      |                 |
    | 30155 | comp_elec   |                                                                                                      |                 |
    | 30156 | comp_elec   |                                                                                                      |                 |
    | 30157 | comp_elec   |                                                                                                      |                 |
    | 30158 | comp_elec   |                                                                                                      |                 |
    | 30159 | comp_elec   |                                                                                                      |                 |
    | 30160 | comp_elec   |                                                                                                      |                 |
    | 30161 | comp_elec   | xv ms-do                                                                                             |                 |
    | 30162 | comp_elec   |                                                                                                      |                 |
    | 30163 | comp_elec   |                                                                                                      |                 |
    | 30164 | comp_elec   |                                                                                                      |                 |
    | 30165 | comp_elec   |                                                                                                      |                 |
    | 30166 | comp_elec   |                                                                                                      |                 |
    | 30167 | comp_elec   |                                                                                                      |                 |
    | 30168 | comp_elec   |                                                                                                      |                 |
    | 30169 | comp_elec   |                                                                                                      |                 |
    | 30170 | comp_elec   |                                                                                                      |                 |
    | 30171 | comp_elec   |                                                                                                      |                 |
    | 30172 | comp_elec   |                                                                                                      |                 |
    | 30173 | comp_elec   |                                                                                                      |                 |
    | 30174 | comp_elec   |                                                                                                      |                 |
    | 30175 | comp_elec   |                                                                                                      |                 |
    | 30176 | comp_elec   |                                                                                                      |                 |
    | 30177 | comp_elec   |                                                                                                      |                 |
    | 30178 | comp_elec   |                                                                                                      |                 |
    | 30179 | comp_elec   |                                                                                                      |                 |
    | 30180 | comp_elec   |                                                                                                      |                 |
    | 30181 | comp_elec   |                                                                                                      |                 |
    | 30182 | comp_elec   |                                                                                                      |                 |
    | 30183 | comp_elec   |                                                                                                      |                 |
    | 30184 | comp_elec   |                                                                                                      |                 |
    | 30185 | comp_elec   |                                                                                                      |                 |
    | 30186 | comp_elec   |                                                                                                      |                 |
    | 30187 | comp_elec   |                                                                                                      |                 |
    | 30188 | comp_elec   |                                                                                                      |                 |
    | 30189 | comp_elec   |                                                                                                      |                 |
    | 30190 | comp_elec   |                                                                                                      |                 |
    | 30191 | comp_elec   |                                                                                                      |                 |
    | 30192 | comp_elec   |                                                                                                      |                 |
    | 30193 | comp_elec   |                                                                                                      |                 |
    | 30194 | comp_elec   |                                                                                                      |                 |
    | 30195 | comp_elec   |                                                                                                      |                 |
    | 30196 | comp_elec   |                                                                                                      |                 |
    | 30197 | comp_elec   |                                                                                                      |                 |
    | 30198 | comp_elec   |                                                                                                      |                 |
    | 30199 | comp_elec   |                                                                                                      |                 |
    | 30200 | comp_elec   |                                                                                                      |                 |
    | 30201 | comp_elec   |                                                                                                      |                 |
    | 30202 | comp_elec   |                                                                                                      |                 |
    | 30203 | comp_elec   |                                                                                                      |                 |
    | 30204 | comp_elec   |                                                                                                      |                 |
    | 30205 | comp_elec   |                                                                                                      |                 |
    | 30206 | comp_elec   |                                                                                                      |                 |
    | 30207 | comp_elec   |                                                                                                      |                 |
    | 30208 | comp_elec   |                                                                                                      |                 |
    | 30209 | comp_elec   |                                                                                                      |                 |
    | 30210 | comp_elec   |                                                                                                      |                 |
    | 30211 | comp_elec   |                                                                                                      |                 |
    | 30212 | comp_elec   |                                                                                                      |                 |
    | 30213 | comp_elec   |                                                                                                      |                 |
    | 30214 | comp_elec   |                                                                                                      |                 |
    | 30215 | comp_elec   |                                                                                                      |                 |
    | 30216 | comp_elec   |                                                                                                      |                 |
    | 30217 | comp_elec   |                                                                                                      |                 |
    | 30218 | comp_elec   |                                                                                                      |                 |
    | 30219 | comp_elec   | need rgb data save imag                                                                              |                 |
    | 30220 | comp_elec   |                                                                                                      |                 |
    | 30221 | comp_elec   |                                                                                                      |                 |
    | 30222 | comp_elec   |                                                                                                      |                 |
    | 30223 | comp_elec   |                                                                                                      |                 |
    | 30224 | comp_elec   |                                                                                                      |                 |
    | 30225 | comp_elec   |                                                                                                      |                 |
    | 30226 | comp_elec   |                                                                                                      |                 |
    | 30227 | comp_elec   |                                                                                                      |                 |
    | 30228 | comp_elec   |                                                                                                      |                 |
    | 30229 | comp_elec   |                                                                                                      |                 |
    | 30230 | comp_elec   |                                                                                                      |                 |
    | 30231 | comp_elec   |                                                                                                      |                 |
    | 30232 | comp_elec   |                                                                                                      |                 |
    | 30233 | comp_elec   |                                                                                                      |                 |
    | 30234 | comp_elec   | univesa driver                                                                                       |                 |
    | 30235 | comp_elec   |                                                                                                      |                 |
    | 30236 | comp_elec   | xv ms-do                                                                                             |                 |
    | 30237 | comp_elec   |                                                                                                      |                 |
    | 30238 | comp_elec   |                                                                                                      |                 |
    | 30239 | comp_elec   |                                                                                                      |                 |
    | 30240 | comp_elec   |                                                                                                      |                 |
    | 30241 | comp_elec   |                                                                                                      |                 |
    | 30242 | comp_elec   |                                                                                                      |                 |
    | 30243 | comp_elec   |                                                                                                      |                 |
    | 30244 | comp_elec   |                                                                                                      |                 |
    | 30245 | comp_elec   |                                                                                                      |                 |
    | 30246 | comp_elec   |                                                                                                      |                 |
    | 30247 | comp_elec   |                                                                                                      |                 |
    | 30248 | comp_elec   |                                                                                                      |                 |
    | 30249 | comp_elec   |                                                                                                      |                 |
    | 30250 | comp_elec   |                                                                                                      |                 |
    | 30251 | comp_elec   |                                                                                                      |                 |
    | 30252 | comp_elec   |                                                                                                      |                 |
    | 30253 | comp_elec   |                                                                                                      |                 |
    | 30254 | comp_elec   |                                                                                                      |                 |
    | 30255 | comp_elec   |                                                                                                      |                 |
    | 30256 | comp_elec   |                                                                                                      |                 |
    | 30257 | comp_elec   |                                                                                                      |                 |
    | 30258 | comp_elec   |                                                                                                      |                 |
    | 30259 | comp_elec   |                                                                                                      |                 |
    | 30260 | comp_elec   |                                                                                                      |                 |
    | 30261 | comp_elec   |                                                                                                      |                 |
    | 30262 | comp_elec   |                                                                                                      |                 |
    | 30263 | comp_elec   |                                                                                                      |                 |
    | 30264 | comp_elec   |                                                                                                      |                 |
    | 30265 | comp_elec   |                                                                                                      |                 |
    | 30266 | comp_elec   |                                                                                                      |                 |
    | 30267 | comp_elec   |                                                                                                      |                 |
    | 30268 | comp_elec   |                                                                                                      |                 |
    | 30269 | comp_elec   |                                                                                                      |                 |
    | 30270 | comp_elec   |                                                                                                      |                 |
    | 30271 | comp_elec   |                                                                                                      |                 |
    | 30272 | comp_elec   | tri view pov file                                                                                    |                 |
    | 30273 | comp_elec   |                                                                                                      |                 |
    | 30274 | comp_elec   |                                                                                                      |                 |
    | 30275 | comp_elec   |                                                                                                      |                 |
    | 30276 | comp_elec   |                                                                                                      |                 |
    | 30277 | comp_elec   |                                                                                                      |                 |
    | 30278 | comp_elec   |                                                                                                      |                 |
    | 30279 | comp_elec   |                                                                                                      |                 |
    | 30280 | comp_elec   |                                                                                                      |                 |
    | 30281 | comp_elec   |                                                                                                      |                 |
    | 30282 | comp_elec   |                                                                                                      |                 |
    | 30283 | comp_elec   | cornerston dualpag driver want                                                                       |                 |
    | 30284 | comp_elec   |                                                                                                      |                 |
    | 30285 | comp_elec   |                                                                                                      |                 |
    | 30286 | comp_elec   |                                                                                                      |                 |
    | 30287 | comp_elec   |                                                                                                      |                 |
    | 30288 | comp_elec   |                                                                                                      |                 |
    | 30289 | comp_elec   |                                                                                                      |                 |
    | 30290 | comp_elec   |                                                                                                      |                 |
    | 30291 | comp_elec   |                                                                                                      |                 |
    | 30292 | comp_elec   |                                                                                                      |                 |
    | 30293 | comp_elec   |                                                                                                      |                 |
    | 30294 | comp_elec   |                                                                                                      |                 |
    | 30295 | comp_elec   |                                                                                                      |                 |
    | 30296 | comp_elec   |                                                                                                      |                 |
    | 30297 | comp_elec   |                                                                                                      |                 |
    | 30298 | comp_elec   |                                                                                                      |                 |
    | 30299 | comp_elec   |                                                                                                      |                 |
    | 30300 | comp_elec   |                                                                                                      |                 |
    | 30301 | comp_elec   | march cub                                                                                            |                 |
    | 30302 | comp_elec   |                                                                                                      |                 |
    | 30303 | comp_elec   |                                                                                                      |                 |
    | 30304 | comp_elec   |                                                                                                      |                 |
    | 30305 | comp_elec   |                                                                                                      |                 |
    | 30306 | comp_elec   |                                                                                                      |                 |
    | 30307 | comp_elec   |                                                                                                      |                 |
    | 30308 | comp_elec   |                                                                                                      |                 |
    | 30309 | comp_elec   |                                                                                                      |                 |
    | 30310 | comp_elec   |                                                                                                      |                 |
    | 30311 | comp_elec   |                                                                                                      |                 |
    | 30312 | comp_elec   |                                                                                                      |                 |
    | 30313 | comp_elec   |                                                                                                      |                 |
    | 30314 | comp_elec   |                                                                                                      |                 |
    | 30315 | comp_elec   |                                                                                                      |                 |
    | 30316 | comp_elec   |                                                                                                      |                 |
    | 30317 | comp_elec   | xlib  bit display info need                                                                          |                 |
    | 30318 | comp_elec   |                                                                                                      |                 |
    | 30319 | comp_elec   |                                                                                                      |                 |
    | 30320 | comp_elec   |                                                                                                      |                 |
    | 30321 | comp_elec   |                                                                                                      |                 |
    | 30322 | comp_elec   |                                                                                                      |                 |
    | 30323 | comp_elec   |                                                                                                      |                 |
    | 30324 | comp_elec   |                                                                                                      |                 |
    | 30325 | comp_elec   |                                                                                                      |                 |
    | 30326 | comp_elec   |                                                                                                      |                 |
    | 30327 | comp_elec   |                                                                                                      |                 |
    | 30328 | comp_elec   | pov file constructor unixx                                                                           |                 |
    | 30329 | comp_elec   |                                                                                                      |                 |
    | 30330 | comp_elec   |                                                                                                      |                 |
    | 30331 | comp_elec   |                                                                                                      |                 |
    | 30332 | comp_elec   |                                                                                                      |                 |
    | 30333 | comp_elec   |                                                                                                      |                 |
    | 30334 | comp_elec   |                                                                                                      |                 |
    | 30335 | comp_elec   |                                                                                                      |                 |
    | 30336 | comp_elec   |                                                                                                      |                 |
    | 30337 | comp_elec   | look tseng vesa driver                                                                               |                 |
    | 30338 | comp_elec   |                                                                                                      |                 |
    | 30339 | comp_elec   |                                                                                                      |                 |
    | 30340 | comp_elec   |                                                                                                      |                 |
    | 30341 | comp_elec   |                                                                                                      |                 |
    | 30342 | comp_elec   |                                                                                                      |                 |
    | 30343 | comp_elec   |                                                                                                      |                 |
    | 30344 | comp_elec   |                                                                                                      |                 |
    | 30345 | comp_elec   |                                                                                                      |                 |
    | 30346 | comp_elec   |                                                                                                      |                 |
    | 30347 | comp_elec   |                                                                                                      |                 |
    | 30348 | comp_elec   |                                                                                                      |                 |
    | 30349 | comp_elec   |                                                                                                      |                 |
    | 30350 | comp_elec   |                                                                                                      |                 |
    | 30351 | comp_elec   |                                                                                                      |                 |
    | 30352 | comp_elec   |                                                                                                      |                 |
    | 30353 | comp_elec   |                                                                                                      |                 |
    | 30354 | comp_elec   |                                                                                                      |                 |
    | 30355 | comp_elec   |                                                                                                      |                 |
    | 30356 | comp_elec   |                                                                                                      |                 |
    | 30357 | comp_elec   |                                                                                                      |                 |
    | 30358 | comp_elec   |                                                                                                      |                 |
    | 30359 | comp_elec   |                                                                                                      |                 |
    | 30360 | comp_elec   |                                                                                                      |                 |
    | 30361 | comp_elec   |                                                                                                      |                 |
    | 30362 | comp_elec   |                                                                                                      |                 |
    | 30363 | comp_elec   |                                                                                                      |                 |
    | 30364 | comp_elec   |                                                                                                      |                 |
    | 30365 | comp_elec   |                                                                                                      |                 |
    | 30366 | comp_elec   |                                                                                                      |                 |
    | 30367 | comp_elec   |                                                                                                      |                 |
    | 30368 | comp_elec   |                                                                                                      |                 |
    | 30369 | comp_elec   |                                                                                                      |                 |
    | 30370 | comp_elec   |                                                                                                      |                 |
    | 30371 | comp_elec   |                                                                                                      |                 |
    | 30372 | comp_elec   |                                                                                                      |                 |
    | 30373 | comp_elec   |                                                                                                      |                 |
    | 30374 | comp_elec   |                                                                                                      |                 |
    | 30375 | comp_elec   |                                                                                                      |                 |
    | 30376 | comp_elec   |                                                                                                      |                 |
    | 30377 | comp_elec   |                                                                                                      |                 |
    | 30378 | comp_elec   |                                                                                                      |                 |
    | 30379 | comp_elec   |                                                                                                      |                 |
    | 30380 | comp_elec   |                                                                                                      |                 |
    | 30381 | comp_elec   |                                                                                                      |                 |
    | 30382 | comp_elec   |                                                                                                      |                 |
    | 30383 | comp_elec   |                                                                                                      |                 |
    | 30384 | comp_elec   |                                                                                                      |                 |
    | 30385 | comp_elec   |                                                                                                      |                 |
    | 30386 | comp_elec   |                                                                                                      |                 |
    | 30387 | comp_elec   |                                                                                                      |                 |
    | 30388 | comp_elec   |                                                                                                      |                 |
    | 30389 | comp_elec   |                                                                                                      |                 |
    | 30390 | comp_elec   |                                                                                                      |                 |
    | 30391 | comp_elec   |                                                                                                      |                 |
    | 30392 | comp_elec   |                                                                                                      |                 |
    | 30393 | comp_elec   |                                                                                                      |                 |
    | 30394 | comp_elec   |                                                                                                      |                 |
    | 30395 | comp_elec   |                                                                                                      |                 |
    | 30396 | comp_elec   |                                                                                                      |                 |
    | 30397 | comp_elec   |                                                                                                      |                 |
    | 30398 | comp_elec   |                                                                                                      |                 |
    | 30399 | comp_elec   |                                                                                                      |                 |
    | 30400 | comp_elec   |                                                                                                      |                 |
    | 30401 | comp_elec   |                                                                                                      |                 |
    | 30402 | comp_elec   |                                                                                                      |                 |
    | 30403 | comp_elec   |                                                                                                      |                 |
    | 30404 | comp_elec   |                                                                                                      |                 |
    | 30405 | comp_elec   |                                                                                                      |                 |
    | 30406 | comp_elec   |                                                                                                      |                 |
    | 30407 | comp_elec   |                                                                                                      |                 |
    | 30408 | comp_elec   |                                                                                                      |                 |
    | 30409 | comp_elec   |                                                                                                      |                 |
    | 30410 | comp_elec   |                                                                                                      |                 |
    | 30411 | comp_elec   |                                                                                                      |                 |
    | 30412 | comp_elec   |                                                                                                      |                 |
    | 30413 | comp_elec   |                                                                                                      |                 |
    | 30414 | comp_elec   |                                                                                                      |                 |
    | 30415 | comp_elec   |                                                                                                      |                 |
    | 30416 | comp_elec   |                                                                                                      |                 |
    | 30417 | comp_elec   |                                                                                                      |                 |
    | 30418 | comp_elec   |                                                                                                      |                 |
    | 30419 | comp_elec   |                                                                                                      |                 |
    | 30420 | comp_elec   |                                                                                                      |                 |
    | 30421 | comp_elec   |                                                                                                      |                 |
    | 30422 | comp_elec   |                                                                                                      |                 |
    | 30423 | comp_elec   |                                                                                                      |                 |
    | 30424 | comp_elec   |                                                                                                      |                 |
    | 30425 | comp_elec   |                                                                                                      |                 |
    | 30426 | comp_elec   |                                                                                                      |                 |
    | 30427 | comp_elec   |                                                                                                      |                 |
    | 30428 | comp_elec   |                                                                                                      |                 |
    | 30429 | comp_elec   |                                                                                                      |                 |
    | 30430 | comp_elec   |                                                                                                      |                 |
    | 30431 | comp_elec   |                                                                                                      |                 |
    | 30432 | comp_elec   |                                                                                                      |                 |
    | 30433 | comp_elec   |                                                                                                      |                 |
    | 30434 | comp_elec   |                                                                                                      |                 |
    | 30435 | comp_elec   |                                                                                                      |                 |
    | 30436 | comp_elec   |                                                                                                      |                 |
    | 30437 | comp_elec   |                                                                                                      |                 |
    | 30438 | comp_elec   |                                                                                                      |                 |
    | 30439 | comp_elec   |                                                                                                      |                 |
    | 30440 | comp_elec   |                                                                                                      |                 |
    | 30441 | comp_elec   |                                                                                                      |                 |
    | 30442 | comp_elec   |                                                                                                      |                 |
    | 30443 | comp_elec   |                                                                                                      |                 |
    | 30444 | comp_elec   |                                                                                                      |                 |
    | 30445 | comp_elec   |                                                                                                      |                 |
    | 30446 | comp_elec   |                                                                                                      |                 |
    | 30447 | comp_elec   |                                                                                                      |                 |
    | 30448 | comp_elec   |                                                                                                      |                 |
    | 30449 | comp_elec   |                                                                                                      |                 |
    | 30450 | comp_elec   |                                                                                                      |                 |
    | 30451 | comp_elec   |                                                                                                      |                 |
    | 30452 | comp_elec   |                                                                                                      |                 |
    | 30453 | comp_elec   |                                                                                                      |                 |
    | 30454 | comp_elec   |                                                                                                      |                 |
    | 30455 | comp_elec   |                                                                                                      |                 |
    | 30456 | comp_elec   |                                                                                                      |                 |
    | 30457 | comp_elec   |                                                                                                      |                 |
    | 30458 | comp_elec   |                                                                                                      |                 |
    | 30459 | comp_elec   |                                                                                                      |                 |
    | 30460 | comp_elec   |                                                                                                      |                 |
    | 30461 | comp_elec   |                                                                                                      |                 |
    | 30462 | comp_elec   |                                                                                                      |                 |
    | 30463 | comp_elec   |                                                                                                      |                 |
    | 30464 | comp_elec   |                                                                                                      |                 |
    | 30465 | comp_elec   |                                                                                                      |                 |
    | 30466 | comp_elec   |                                                                                                      |                 |
    | 30467 | comp_elec   |                                                                                                      |                 |
    | 30468 | comp_elec   |                                                                                                      |                 |
    | 30469 | comp_elec   |                                                                                                      |                 |
    | 30470 | comp_elec   |                                                                                                      |                 |
    | 30471 | comp_elec   |                                                                                                      |                 |
    | 30472 | comp_elec   | front end povray                                                                                     |                 |
    | 30473 | comp_elec   |                                                                                                      |                 |
    | 30474 | comp_elec   |                                                                                                      |                 |
    | 30475 | comp_elec   |                                                                                                      |                 |
    | 30476 | comp_elec   |                                                                                                      |                 |
    | 30477 | comp_elec   |                                                                                                      |                 |
    | 30478 | comp_elec   |                                                                                                      |                 |
    | 30479 | comp_elec   |                                                                                                      |                 |
    | 30480 | comp_elec   |                                                                                                      |                 |
    | 30481 | comp_elec   |                                                                                                      |                 |
    | 30482 | comp_elec   |                                                                                                      |                 |
    | 30483 | comp_elec   |                                                                                                      |                 |
    | 30484 | comp_elec   |                                                                                                      |                 |
    | 30485 | comp_elec   |                                                                                                      |                 |
    | 30486 | comp_elec   |                                                                                                      |                 |
    | 30487 | comp_elec   |                                                                                                      |                 |
    | 30488 | comp_elec   |                                                                                                      |                 |
    | 30489 | comp_elec   |                                                                                                      |                 |
    | 30490 | comp_elec   |                                                                                                      |                 |
    | 30491 | comp_elec   |                                                                                                      |                 |
    | 30492 | comp_elec   |                                                                                                      |                 |
    | 30493 | comp_elec   |                                                                                                      |                 |
    | 30494 | comp_elec   |                                                                                                      |                 |
    | 30495 | comp_elec   |                                                                                                      |                 |
    | 30496 | comp_elec   |                                                                                                      |                 |
    | 30497 | comp_elec   |                                                                                                      |                 |
    | 30498 | comp_elec   |                                                                                                      |                 |
    | 30499 | comp_elec   | dxf pcx gif tif tga                                                                                  |                 |
    | 30500 | comp_elec   |                                                                                                      |                 |
    | 30501 | comp_elec   |                                                                                                      |                 |
    | 30502 | comp_elec   |                                                                                                      |                 |
    | 30503 | comp_elec   |                                                                                                      |                 |
    | 30504 | comp_elec   |                                                                                                      |                 |
    | 30505 | comp_elec   |                                                                                                      |                 |
    | 30506 | comp_elec   |                                                                                                      |                 |
    | 30507 | comp_elec   | dna helix                                                                                            |                 |
    | 30508 | comp_elec   |                                                                                                      |                 |
    | 30509 | comp_elec   |                                                                                                      |                 |
    | 30510 | comp_elec   |                                                                                                      |                 |
    | 30511 | comp_elec   |                                                                                                      |                 |
    | 30512 | comp_elec   |                                                                                                      |                 |
    | 30513 | comp_elec   |                                                                                                      |                 |
    | 30514 | comp_elec   |                                                                                                      |                 |
    | 30515 | comp_elec   |                                                                                                      |                 |
    | 30516 | comp_elec   |                                                                                                      |                 |
    | 30517 | comp_elec   |                                                                                                      |                 |
    | 30518 | comp_elec   |                                                                                                      |                 |
    | 30519 | comp_elec   |                                                                                                      |                 |
    | 30520 | comp_elec   |                                                                                                      |                 |
    | 30521 | comp_elec   |                                                                                                      |                 |
    | 30522 | comp_elec   | povray tga - rle                                                                                     |                 |
    | 30523 | comp_elec   |                                                                                                      |                 |
    | 30524 | comp_elec   |                                                                                                      |                 |
    | 30525 | comp_elec   |                                                                                                      |                 |
    | 30526 | comp_elec   | virtual realiti x cheap                                                                              |                 |
    | 30527 | comp_elec   |                                                                                                      |                 |
    | 30528 | comp_elec   |                                                                                                      |                 |
    | 30529 | comp_elec   |                                                                                                      |                 |
    | 30530 | comp_elec   |                                                                                                      |                 |
    | 30531 | comp_elec   |                                                                                                      |                 |
    | 30532 | comp_elec   |                                                                                                      |                 |
    | 30533 | comp_elec   | sourc codehelp ip packag pleas                                                                       |                 |
    | 30534 | comp_elec   |                                                                                                      |                 |
    | 30535 | comp_elec   |                                                                                                      |                 |
    | 30536 | comp_elec   |                                                                                                      |                 |
    | 30537 | comp_elec   |                                                                                                      |                 |
    | 30538 | comp_elec   |                                                                                                      |                 |
    | 30539 | comp_elec   |                                                                                                      |                 |
    | 30540 | comp_elec   |                                                                                                      |                 |
    | 30541 | comp_elec   |                                                                                                      |                 |
    | 30542 | comp_elec   |                                                                                                      |                 |
    | 30543 | comp_elec   |                                                                                                      |                 |
    | 30544 | comp_elec   |                                                                                                      |                 |
    | 30545 | comp_elec   |                                                                                                      |                 |
    | 30546 | comp_elec   |                                                                                                      |                 |
    | 30547 | comp_elec   |                                                                                                      |                 |
    | 30548 | comp_elec   |                                                                                                      |                 |
    | 30549 | comp_elec   |                                                                                                      |                 |
    | 30550 | comp_elec   |                                                                                                      |                 |
    | 30551 | comp_elec   |                                                                                                      |                 |
    | 30552 | comp_elec   |                                                                                                      |                 |
    | 30553 | comp_elec   |                                                                                                      |                 |
    | 30554 | comp_elec   |                                                                                                      |                 |
    | 30555 | comp_elec   |                                                                                                      |                 |
    | 30556 | comp_elec   |                                                                                                      |                 |
    | 30557 | comp_elec   |                                                                                                      |                 |
    | 30558 | comp_elec   |                                                                                                      |                 |
    | 30559 | comp_elec   |                                                                                                      |                 |
    | 30560 | comp_elec   |                                                                                                      |                 |
    | 30561 | comp_elec   |                                                                                                      |                 |
    | 30562 | comp_elec   |                                                                                                      |                 |
    | 30563 | comp_elec   |                                                                                                      |                 |
    | 30564 | comp_elec   |                                                                                                      |                 |
    | 30565 | comp_elec   |                                                                                                      |                 |
    | 30566 | comp_elec   |                                                                                                      |                 |
    | 30567 | comp_elec   |                                                                                                      |                 |
    | 30568 | comp_elec   |                                                                                                      |                 |
    | 30569 | comp_elec   |                                                                                                      |                 |
    | 30570 | comp_elec   |                                                                                                      |                 |
    | 30571 | comp_elec   |                                                                                                      |                 |
    | 30572 | comp_elec   |                                                                                                      |                 |
    | 30573 | comp_elec   |                                                                                                      |                 |
    | 30574 | comp_elec   |                                                                                                      |                 |
    | 30575 | comp_elec   |                                                                                                      |                 |
    | 30576 | comp_elec   |                                                                                                      |                 |
    | 30577 | comp_elec   |                                                                                                      |                 |
    | 30578 | comp_elec   |                                                                                                      |                 |
    | 30579 | comp_elec   |                                                                                                      |                 |
    | 30580 | comp_elec   |                                                                                                      |                 |
    | 30581 | comp_elec   |                                                                                                      |                 |
    | 30582 | comp_elec   |                                                                                                      |                 |
    | 30583 | comp_elec   |                                                                                                      |                 |
    | 30584 | comp_elec   |                                                                                                      |                 |
    | 30585 | comp_elec   |                                                                                                      |                 |
    | 30586 | comp_elec   |                                                                                                      |                 |
    | 30587 | comp_elec   |                                                                                                      |                 |
    | 30588 | comp_elec   |                                                                                                      |                 |
    | 30589 | comp_elec   |                                                                                                      |                 |
    | 30590 | comp_elec   |                                                                                                      |                 |
    | 30591 | comp_elec   |                                                                                                      |                 |
    | 30592 | comp_elec   |                                                                                                      |                 |
    | 30593 | comp_elec   | phig user group confer                                                                               |                 |
    | 30594 | comp_elec   |                                                                                                      |                 |
    | 30595 | comp_elec   |                                                                                                      |                 |
    | 30596 | comp_elec   |                                                                                                      |                 |
    | 30597 | comp_elec   |                                                                                                      |                 |
    | 30598 | comp_elec   |                                                                                                      |                 |
    | 30599 | comp_elec   |                                                                                                      |                 |
    | 30600 | comp_elec   |                                                                                                      |                 |
    | 30601 | comp_elec   |                                                                                                      |                 |
    | 30602 | comp_elec   |                                                                                                      |                 |
    | 30603 | comp_elec   |                                                                                                      |                 |
    | 30604 | comp_elec   |                                                                                                      |                 |
    | 30605 | comp_elec   |                                                                                                      |                 |
    | 30606 | comp_elec   |                                                                                                      |                 |
    | 30607 | comp_elec   |                                                                                                      |                 |
    | 30608 | comp_elec   |                                                                                                      |                 |
    | 30609 | comp_elec   |                                                                                                      |                 |
    | 30610 | comp_elec   |                                                                                                      |                 |
    | 30611 | comp_elec   |                                                                                                      |                 |
    | 30612 | comp_elec   |                                                                                                      |                 |
    | 30613 | comp_elec   |                                                                                                      |                 |
    | 30614 | comp_elec   |                                                                                                      |                 |
    | 30615 | comp_elec   |                                                                                                      |                 |
    | 30616 | comp_elec   |                                                                                                      |                 |
    | 30617 | comp_elec   |                                                                                                      |                 |
    | 30618 | comp_elec   |                                                                                                      |                 |
    | 30619 | comp_elec   |                                                                                                      |                 |
    | 30620 | comp_elec   |                                                                                                      |                 |
    | 30621 | comp_elec   |                                                                                                      |                 |
    | 30622 | comp_elec   |                                                                                                      |                 |
    | 30623 | comp_elec   |                                                                                                      |                 |
    | 30624 | comp_elec   |                                                                                                      |                 |
    | 30625 | comp_elec   |                                                                                                      |                 |
    | 30626 | comp_elec   | pbm dec whensthenewvers                                                                              |                 |
    | 30627 | comp_elec   |                                                                                                      |                 |
    | 30628 | comp_elec   |                                                                                                      |                 |
    | 30629 | comp_elec   |                                                                                                      |                 |
    | 30630 | comp_elec   |                                                                                                      |                 |
    | 30631 | comp_elec   |                                                                                                      |                 |
    | 30632 | comp_elec   |                                                                                                      |                 |
    | 30633 | comp_elec   |                                                                                                      |                 |
    | 30634 | comp_elec   |                                                                                                      |                 |
    | 30635 | comp_elec   |                                                                                                      |                 |
    | 30636 | comp_elec   |                                                                                                      |                 |
    | 30637 | comp_elec   |                                                                                                      |                 |
    | 30638 | comp_elec   |                                                                                                      |                 |
    | 30639 | comp_elec   |                                                                                                      |                 |
    | 30640 | comp_elec   |                                                                                                      |                 |
    | 30641 | comp_elec   |                                                                                                      |                 |
    | 30642 | comp_elec   |                                                                                                      |                 |
    | 30643 | comp_elec   |                                                                                                      |                 |
    | 30644 | comp_elec   | tiff complex                                                                                         |                 |
    | 30645 | comp_elec   |                                                                                                      |                 |
    | 30646 | comp_elec   |                                                                                                      |                 |
    | 30647 | comp_elec   |                                                                                                      |                 |
    | 30648 | comp_elec   |                                                                                                      |                 |
    | 30649 | comp_elec   |                                                                                                      |                 |
    | 30650 | comp_elec   |                                                                                                      |                 |
    | 30651 | comp_elec   |                                                                                                      |                 |
    | 30652 | comp_elec   |                                                                                                      |                 |
    | 30653 | comp_elec   |                                                                                                      |                 |
    | 30654 | comp_elec   |                                                                                                      |                 |
    | 30655 | comp_elec   |                                                                                                      |                 |
    | 30656 | comp_elec   |                                                                                                      |                 |
    | 30657 | comp_elec   |                                                                                                      |                 |
    | 30658 | comp_elec   |                                                                                                      |                 |
    | 30659 | comp_elec   |                                                                                                      |                 |
    | 30660 | comp_elec   |                                                                                                      |                 |
    | 30661 | comp_elec   |                                                                                                      |                 |
    | 30662 | comp_elec   |                                                                                                      |                 |
    | 30663 | comp_elec   |                                                                                                      |                 |
    | 30664 | comp_elec   |                                                                                                      |                 |
    | 30665 | comp_elec   |                                                                                                      |                 |
    | 30666 | comp_elec   |                                                                                                      |                 |
    | 30667 | comp_elec   |                                                                                                      |                 |
    | 30668 | comp_elec   |                                                                                                      |                 |
    | 30669 | comp_elec   |                                                                                                      |                 |
    | 30670 | comp_elec   |                                                                                                      |                 |
    | 30671 | comp_elec   |                                                                                                      |                 |
    | 30672 | comp_elec   |                                                                                                      |                 |
    | 30673 | comp_elec   |                                                                                                      |                 |
    | 30674 | comp_elec   |                                                                                                      |                 |
    | 30675 | comp_elec   |                                                                                                      |                 |
    | 30676 | comp_elec   |                                                                                                      |                 |
    | 30677 | comp_elec   |                                                                                                      |                 |
    | 30678 | comp_elec   |                                                                                                      |                 |
    | 30679 | comp_elec   |                                                                                                      |                 |
    | 30680 | comp_elec   |                                                                                                      |                 |
    | 30681 | comp_elec   |                                                                                                      |                 |
    | 30682 | comp_elec   | gif targa                                                                                            |                 |
    | 30683 | comp_elec   |                                                                                                      |                 |
    | 30684 | comp_elec   |                                                                                                      |                 |
    | 30685 | comp_elec   |                                                                                                      |                 |
    | 30686 | comp_elec   |                                                                                                      |                 |
    | 30687 | comp_elec   |                                                                                                      |                 |
    | 30688 | comp_elec   |                                                                                                      |                 |
    | 30689 | comp_elec   |                                                                                                      |                 |
    | 30690 | comp_elec   |                                                                                                      |                 |
    | 30691 | comp_elec   |                                                                                                      |                 |
    | 30692 | comp_elec   |                                                                                                      |                 |
    | 30693 | comp_elec   |                                                                                                      |                 |
    | 30694 | comp_elec   |                                                                                                      |                 |
    | 30695 | comp_elec   |                                                                                                      |                 |
    | 30696 | comp_elec   |                                                                                                      |                 |
    | 30697 | comp_elec   |                                                                                                      |                 |
    | 30698 | comp_elec   |                                                                                                      |                 |
    | 30699 | comp_elec   |                                                                                                      |                 |
    | 30700 | comp_elec   | fractal terrain gener                                                                                |                 |
    | 30701 | comp_elec   |                                                                                                      |                 |
    | 30702 | comp_elec   |                                                                                                      |                 |
    | 30703 | comp_elec   |                                                                                                      |                 |
    | 30704 | comp_elec   |                                                                                                      |                 |
    | 30705 | comp_elec   |                                                                                                      |                 |
    | 30706 | comp_elec   |                                                                                                      |                 |
    | 30707 | comp_elec   |                                                                                                      |                 |
    | 30708 | comp_elec   |                                                                                                      |                 |
    | 30709 | comp_elec   |                                                                                                      |                 |
    | 30710 | comp_elec   |                                                                                                      |                 |
    | 30711 | comp_elec   |                                                                                                      |                 |
    | 30712 | comp_elec   |                                                                                                      |                 |
    | 30713 | comp_elec   |                                                                                                      |                 |
    | 30714 | comp_elec   |                                                                                                      |                 |
    | 30715 | comp_elec   |                                                                                                      |                 |
    | 30716 | comp_elec   |                                                                                                      |                 |
    | 30717 | comp_elec   |                                                                                                      |                 |
    | 30718 | comp_elec   |                                                                                                      |                 |
    | 30719 | comp_elec   |                                                                                                      |                 |
    | 30720 | comp_elec   |                                                                                                      |                 |
    | 30721 | comp_elec   |                                                                                                      |                 |
    | 30722 | comp_elec   |                                                                                                      |                 |
    | 30723 | comp_elec   |                                                                                                      |                 |
    | 30724 | comp_elec   |                                                                                                      |                 |
    | 30725 | comp_elec   |                                                                                                      |                 |
    | 30726 | comp_elec   |                                                                                                      |                 |
    | 30727 | comp_elec   |                                                                                                      |                 |
    | 30728 | comp_elec   |                                                                                                      |                 |
    | 30729 | comp_elec   |                                                                                                      |                 |
    | 30730 | comp_elec   |                                                                                                      |                 |
    | 30731 | comp_elec   |                                                                                                      |                 |
    | 30732 | comp_elec   |                                                                                                      |                 |
    | 30733 | comp_elec   |                                                                                                      |                 |
    | 30734 | comp_elec   |                                                                                                      |                 |
    | 30735 | comp_elec   |                                                                                                      |                 |
    | 30736 | comp_elec   |                                                                                                      |                 |
    | 30737 | comp_elec   |                                                                                                      |                 |
    | 30738 | comp_elec   |                                                                                                      |                 |
    | 30739 | comp_elec   |                                                                                                      |                 |
    | 30740 | comp_elec   |                                                                                                      |                 |
    | 30741 | comp_elec   |                                                                                                      |                 |
    | 30742 | comp_elec   |                                                                                                      |                 |
    | 30743 | comp_elec   |                                                                                                      |                 |
    | 30744 | comp_elec   |                                                                                                      |                 |
    | 30745 | comp_elec   |                                                                                                      |                 |
    | 30746 | comp_elec   |                                                                                                      |                 |
    | 30747 | comp_elec   |                                                                                                      |                 |
    | 30748 | comp_elec   |                                                                                                      |                 |
    | 30749 | comp_elec   |                                                                                                      |                 |
    | 30750 | comp_elec   |                                                                                                      |                 |
    | 30751 | comp_elec   |                                                                                                      |                 |
    | 30752 | comp_elec   |                                                                                                      |                 |
    | 30753 | comp_elec   |                                                                                                      |                 |
    | 30754 | comp_elec   |                                                                                                      |                 |
    | 30755 | comp_elec   |                                                                                                      |                 |
    | 30756 | comp_elec   |                                                                                                      |                 |
    | 30757 | comp_elec   |                                                                                                      |                 |
    | 30758 | comp_elec   |                                                                                                      |                 |
    | 30759 | comp_elec   |                                                                                                      |                 |
    | 30760 | comp_elec   |                                                                                                      |                 |
    | 30761 | comp_elec   |                                                                                                      |                 |
    | 30762 | comp_elec   |                                                                                                      |                 |
    | 30763 | comp_elec   |                                                                                                      |                 |
    | 30764 | comp_elec   |                                                                                                      |                 |
    | 30765 | comp_elec   |                                                                                                      |                 |
    | 30766 | comp_elec   |                                                                                                      |                 |
    | 30767 | comp_elec   |                                                                                                      |                 |
    | 30768 | comp_elec   |                                                                                                      |                 |
    | 30769 | comp_elec   |                                                                                                      |                 |
    | 30770 | comp_elec   |                                                                                                      |                 |
    | 30771 | comp_elec   |                                                                                                      |                 |
    | 30772 | comp_elec   |                                                                                                      |                 |
    | 30773 | comp_elec   |                                                                                                      |                 |
    | 30774 | comp_elec   |                                                                                                      |                 |
    | 30775 | comp_elec   |                                                                                                      |                 |
    | 30776 | comp_elec   |                                                                                                      |                 |
    | 30777 | comp_elec   |                                                                                                      |                 |
    | 30778 | comp_elec   |                                                                                                      |                 |
    | 30779 | comp_elec   |                                                                                                      |                 |
    | 30780 | comp_elec   |                                                                                                      |                 |
    | 30781 | comp_elec   |                                                                                                      |                 |
    | 30782 | comp_elec   |                                                                                                      |                 |
    | 30783 | comp_elec   |                                                                                                      |                 |
    | 30784 | comp_elec   |                                                                                                      |                 |
    | 30785 | comp_elec   |                                                                                                      |                 |
    | 30786 | comp_elec   |                                                                                                      |                 |
    | 30787 | comp_elec   |                                                                                                      |                 |
    | 30788 | comp_elec   |                                                                                                      |                 |
    | 30789 | comp_elec   |                                                                                                      |                 |
    | 30790 | comp_elec   |                                                                                                      |                 |
    | 30791 | comp_elec   |                                                                                                      |                 |
    | 30792 | comp_elec   |                                                                                                      |                 |
    | 30793 | comp_elec   |                                                                                                      |                 |
    | 30794 | comp_elec   |                                                                                                      |                 |
    | 30795 | comp_elec   |                                                                                                      |                 |
    | 30796 | comp_elec   |                                                                                                      |                 |
    | 30797 | comp_elec   |                                                                                                      |                 |
    | 30798 | comp_elec   |                                                                                                      |                 |
    | 30799 | comp_elec   |                                                                                                      |                 |
    | 30800 | comp_elec   |                                                                                                      |                 |
    | 30801 | comp_elec   |                                                                                                      |                 |
    | 30802 | comp_elec   |                                                                                                      |                 |
    | 30803 | comp_elec   |                                                                                                      |                 |
    | 30804 | comp_elec   |                                                                                                      |                 |
    | 30805 | comp_elec   |                                                                                                      |                 |
    | 30806 | comp_elec   |                                                                                                      |                 |
    | 30807 | comp_elec   |                                                                                                      |                 |
    | 30808 | comp_elec   |                                                                                                      |                 |
    | 30809 | comp_elec   |                                                                                                      |                 |
    | 30810 | comp_elec   |                                                                                                      |                 |
    | 30811 | comp_elec   |                                                                                                      |                 |
    | 30812 | comp_elec   |                                                                                                      |                 |
    | 30813 | comp_elec   |                                                                                                      |                 |
    | 30814 | comp_elec   |                                                                                                      |                 |
    | 30815 | comp_elec   |                                                                                                      |                 |
    | 30816 | comp_elec   |                                                                                                      |                 |
    | 30817 | comp_elec   |                                                                                                      |                 |
    | 30837 | comp_elec   | hiraganakatakana tt font newsgroup                                                                   |                 |
    | 30861 | comp_elec   | redirect print manag file hpcc                                                                       |                 |
    | 30869 | comp_elec   | cool bmp file articl apr dlss jame dlss jame cum write newsgroup                                     |                 |
    | 30895 | comp_elec   | codabar font hi need codabar font win tt thank                                                       |                 |
    | 30947 | comp_elec   | redirect print manag file  hpcccorphpcom reed hpcccorphpcom perri reed write hpcc                    |                 |
    | 30976 | comp_elec   | hp laser jet l window ha anybodi chanc find new hp laser jet l behav window daniel royer             |                 |
    | 31004 | comp_elec   | need icon printer util look printer util stay window  icon let drag file issu print                  |                 |
    | 31018 | comp_elec   | nt readi abl call favorit mail order softwar shop buy nt jeff dragovich dragov cevaxceuiucedu        |                 |
    | 31036 | comp_elec   | ani util let remap keyboard ms win ani util let remap keyboard ms win thank ani pointer              |                 |
    | 31044 | comp_elec   | bit rgb work  bit rgb bmp file need comvert   bit imag convert  bit imag   bit rgb imag thank        |                 |
    | 31054 | comp_elec   | find diamond speedstar x driver cicaindianaedu pcdriver current version                              |                 |
    | 31096 | comp_elec   | vbrundll someon point direct thi file thank come visual basic new version vbrundll thx dave l        |                 |
    | 31100 | comp_elec   | bit access gateway dxv doe anyon know csn ca nt duse  bit access                                     |                 |
    | 31136 | comp_elec   | anyon heard deltree brandon wise bwise nyxcsduedu                                                    |                 |
    | 31196 | comp_elec   | instal printshop delux window norton desktop                                                         |                 |
    | 31201 | comp_elec   | font size window xx                                                                                  |                 |
    | 31203 | comp_elec   | challeng microsoft support get pictur find humor run window  app   make os credibl cliff             |                 |
    | 31266 | comp_elec   | permana swap file  dbldisk                                                                           |                 |
    | 31289 | comp_elec   | ati ultra pro driver doe anybodi know ftp site latest window driver ati gup thank                    |                 |
    | 31302 | comp_elec   | speedstar vga card win driver driver updat avail directli diamond even ship charg least              |                 |
    | 31307 | comp_elec   | panason kx-pi driver doe anyon know print driver window panason kx-pi -pin dot matrix printer        |                 |
    | 31309 | comp_elec   | help winqvt similar problem - tri chang netmask   tommi                                              |                 |
    | 31403 | comp_elec   | ftp tool window ani one know ftp tool window get tool thank ani help hj                              |                 |
    | 31451 | comp_elec   | window  slower use  adam benson mt pearl nf adamb garfieldcsmunca                                    |                 |
    | 31472 | comp_elec   | util updat winini systemini nead util updat delet ad chang ini file window find ani ftp host svein   |                 |
    | 31519 | comp_elec   | window  slower use  system frank calloway                                                            |                 |
    | 31526 | comp_elec   | protec mode look inform w-nt use protec mode hw support                                              |                 |
    | 31543 | comp_elec   | cirru logic  graph card                                                                              |                 |
    | 31571 | comp_elec   | printer secur mark juric ai program mjuric aiugaedu univers georgia athen georgia                    |                 |
    | 31572 | comp_elec   | cica mirror find alway almost anyway busi dial tri repeatedli usual onli   tri alway get connect     |                 |
    | 31601 | comp_elec   | trumpet window news reader ftp site trumpet wuarchiv umich someth beaten path thank malcolm          |                 |
    | 31616 | comp_elec   | adaptec scsi devic driver win                                                                        |                 |
    | 31641 | comp_elec   | trident vga driver hi trident tvga- video card need updat driver win get ftp site thank bj           |                 |
    | 31682 | comp_elec   | canon bubl jet printer hi somebodi tell much canon bj buy cheapest price thank advanc --  -- -- --   |                 |
    | 31688 | comp_elec   | program manag kill group file ani clue time enter win  progman say need rebuild group quit annoy     |                 |
    | 31697 | comp_elec   | cica mirror tri wuarchivewustledu mirrorswin directori                                               |                 |
    | 31715 | comp_elec   | protect fault hi wa wonder anyon could help error messag goe doe mean run ms window  thank advanc -- |                 |
    | 31744 | comp_elec   | program manag two question actual sever sharwar util cn chang fav plug-in bryan dunn                 |                 |
    | 31765 | comp_elec   | summari borlandmicrosoft databas c librari could post descript objectbas chosen product thank        |                 |
    | 31775 | comp_elec   | winqvtnet ndi token ring                                                                             |                 |
    | 31822 | comp_elec   | hiraganakatakana tt font newsgroup                                                                   |                 |
    | 31846 | comp_elec   | redirect print manag file hpcc                                                                       |                 |
    | 31854 | comp_elec   | cool bmp file articl apr dlss jame dlss jame cum write newsgroup                                     |                 |
    | 31880 | comp_elec   | codabar font hi need codabar font win tt thank                                                       |                 |
    | 31932 | comp_elec   | redirect print manag file  hpcccorphpcom reed hpcccorphpcom perri reed write hpcc                    |                 |
    | 31961 | comp_elec   | hp laser jet l window ha anybodi chanc find new hp laser jet l behav window daniel royer             |                 |
    | 31989 | comp_elec   | need icon printer util look printer util stay window  icon let drag file issu print                  |                 |
    | 32003 | comp_elec   | nt readi abl call favorit mail order softwar shop buy nt jeff dragovich dragov cevaxceuiucedu        |                 |
    | 32021 | comp_elec   | ani util let remap keyboard ms win ani util let remap keyboard ms win thank ani pointer              |                 |
    | 32029 | comp_elec   | bit rgb work  bit rgb bmp file need comvert   bit imag convert  bit imag   bit rgb imag thank        |                 |
    | 32039 | comp_elec   | find diamond speedstar x driver cicaindianaedu pcdriver current version                              |                 |
    | 32081 | comp_elec   | vbrundll someon point direct thi file thank come visual basic new version vbrundll thx dave l        |                 |
    | 32085 | comp_elec   | bit access gateway dxv doe anyon know csn ca nt duse  bit access                                     |                 |
    | 32121 | comp_elec   | anyon heard deltree brandon wise bwise nyxcsduedu                                                    |                 |
    | 32180 | comp_elec   |                                                                                                      |                 |
    | 32181 | comp_elec   | instal printshop delux window norton desktop                                                         |                 |
    | 32186 | comp_elec   | font size window xx                                                                                  |                 |
    | 32188 | comp_elec   | challeng microsoft support get pictur find humor run window  app   make os credibl cliff             |                 |
    | 32251 | comp_elec   | permana swap file  dbldisk                                                                           |                 |
    | 32274 | comp_elec   | ati ultra pro driver doe anybodi know ftp site latest window driver ati gup thank                    |                 |
    | 32287 | comp_elec   | speedstar vga card win driver driver updat avail directli diamond even ship charg least              |                 |
    | 32292 | comp_elec   | panason kx-pi driver doe anyon know print driver window panason kx-pi -pin dot matrix printer        |                 |
    | 32294 | comp_elec   | help winqvt similar problem - tri chang netmask   tommi                                              |                 |
    | 32388 | comp_elec   | ftp tool window ani one know ftp tool window get tool thank ani help hj                              |                 |
    | 32436 | comp_elec   | window  slower use  adam benson mt pearl nf adamb garfieldcsmunca                                    |                 |
    | 32457 | comp_elec   | util updat winini systemini nead util updat delet ad chang ini file window find ani ftp host svein   |                 |
    | 32504 | comp_elec   | window  slower use  system frank calloway                                                            |                 |
    | 32511 | comp_elec   | protec mode look inform w-nt use protec mode hw support                                              |                 |
    | 32528 | comp_elec   | cirru logic  graph card                                                                              |                 |
    | 32556 | comp_elec   | printer secur mark juric ai program mjuric aiugaedu univers georgia athen georgia                    |                 |
    | 32557 | comp_elec   | cica mirror find alway almost anyway busi dial tri repeatedli usual onli   tri alway get connect     |                 |
    | 32586 | comp_elec   | trumpet window news reader ftp site trumpet wuarchiv umich someth beaten path thank malcolm          |                 |
    | 32601 | comp_elec   | adaptec scsi devic driver win                                                                        |                 |
    | 32626 | comp_elec   | trident vga driver hi trident tvga- video card need updat driver win get ftp site thank bj           |                 |
    | 32673 | comp_elec   | program manag kill group file ani clue time enter win  progman say need rebuild group quit annoy     |                 |
    | 32682 | comp_elec   | cica mirror tri wuarchivewustledu mirrorswin directori                                               |                 |
    | 32729 | comp_elec   | program manag two question actual sever sharwar util cn chang fav plug-in bryan dunn                 |                 |
    | 32750 | comp_elec   | summari borlandmicrosoft databas c librari could post descript objectbas chosen product thank        |                 |
    | 32760 | comp_elec   | winqvtnet ndi token ring                                                                             |                 |
    | 32789 | comp_elec   | flame therapi think would great idea new group creat compsysibmpcflametherapi anybodi agre           |                 |
    | 32842 | comp_elec   | look book hi netter look book show fix hardwar problem pleas let know ani book mind thank            |                 |
    | 32849 | comp_elec   | wen  monitor help doe anybodi ani info thi monitor manufactur help e-mail pleas scotti               |                 |
    | 32899 | comp_elec   | refresh rate nec fgx someon tell maximum horizont vertic refresh rate nec fgx fge thank              |                 |
    | 32930 | comp_elec   | com port modem mous conflict -realli articl apr wnbbsnbgsuborg                                       |                 |
    | 32947 | comp_elec   | --- western digit repli --- western digit voic mail - get inform mani drive actual person end        |                 |
    | 32999 | comp_elec   | amd cpu ani comment amd microprocessor good bad thank pat                                            |                 |
    | 33021 | comp_elec   | toshiba b cd-rom ani problem pas toshiba  combo problem                                              |                 |
    | 33044 | comp_elec   | help doc motherboard                                                                                 |                 |
    | 33051 | comp_elec   | maxtor drive geometryjump                                                                            |                 |
    | 33078 | comp_elec   | address interliv address interliv memmori modul interliv thank advanc info robert                    |                 |
    | 33109 | comp_elec   | mb fd want subject say pleas email soon skcgoh tartarusuwaeduau                                      |                 |
    | 33170 | comp_elec   | perfect mag mxf monitor articl   last newsgroup                                                      |                 |
    | 33214 | comp_elec   | cga cardmonitor want titl say whi lee lee tosspotsvcom                                               |                 |
    | 33221 | comp_elec   | fastmicro busi heard fastmicro went busi thi true nt answer  number --                               |                 |
    | 33268 | comp_elec   | umbdrzip ani later version                                                                           |                 |
    | 33292 | comp_elec   | eric bosco eric send email address lost reconsid kevin                                               |                 |
    | 33343 | comp_elec   | diamond stealth  give  winmark dx stealth  vlb get  winmark ver                                      |                 |
    | 33348 | comp_elec   | test nt read -- gosh think instal viru wa call ms dos nt copi floppi burn love window crash          |                 |
    | 33354 | comp_elec   | smc e arcnet car sale ea  new smc e arcnet card sale brand new  wow cupportalcom walli waggon        |                 |
    | 33355 | comp_elec   | hay jt fax card sale  like new hay jt fax sale  offer trade walli waggon wow cupportalcom            |                 |
    | 33429 | comp_elec   | dx math co-pro vs dx  cpu give better perform math intens program - dx - dx thank advanc chri teagu  |                 |
    | 33468 | comp_elec   | wincim  baud similar problem download use wincim discov disabl data compress modem work fine         |                 |
    | 33489 | comp_elec   | diamond stealth help articl   last newsgroup                                                         |                 |
    | 33491 | comp_elec   | hot cpu articl   last newsgroup                                                                      |                 |
    | 33501 | comp_elec   | need videotap pc output need videotap copi pc pd program pleas let know thi marc dna ucsusledu       |                 |
    | 33505 | comp_elec   | monitor - nanao hello follow discuss  monitor                                                        |                 |
    | 33566 | comp_elec   | monitor - nanao johnn eskimocom john navitski write hello follow discuss  monitor                    |                 |
    | 33676 | comp_elec   | help led connector motherboard articl   last newsgroup                                               |                 |
    | 33677 | comp_elec   | help ide drive instal problem articl   last newsgroup                                                |                 |
    | 33682 | comp_elec   | consum warn midwest micro ohio midwest micro articl   last newsgroup compdcommodem                   |                 |
    | 33685 | comp_elec   | dx vs dx articl   last newsgroup                                                                     |                 |
    | 33698 | comp_elec   | isa ca nt use  meg ram ok comput liter done ram  meg isa machin pleas e-mail thank advanc charley    |                 |
    | 33759 | comp_elec   | monitor - nanao in-reply-to johnn eskimocom messag  apr   gmt newsgroup                              |                 |
    | 33771 | comp_elec   | flame therapi think would great idea new group creat compsysibmpcflametherapi anybodi agre           |                 |
    | 33824 | comp_elec   | look book hi netter look book show fix hardwar problem pleas let know ani book mind thank            |                 |
    | 33831 | comp_elec   | wen  monitor help doe anybodi ani info thi monitor manufactur help e-mail pleas scotti               |                 |
    | 33881 | comp_elec   | refresh rate nec fgx someon tell maximum horizont vertic refresh rate nec fgx fge thank              |                 |
    | 33912 | comp_elec   | com port modem mous conflict -realli articl apr wnbbsnbgsuborg                                       |                 |
    | 33929 | comp_elec   | --- western digit repli --- western digit voic mail - get inform mani drive actual person end        |                 |
    | 33981 | comp_elec   | amd cpu ani comment amd microprocessor good bad thank pat                                            |                 |
    | 34003 | comp_elec   | toshiba b cd-rom ani problem pas toshiba  combo problem                                              |                 |
    | 34026 | comp_elec   | help doc motherboard                                                                                 |                 |
    | 34033 | comp_elec   | maxtor drive geometryjump                                                                            |                 |
    | 34060 | comp_elec   | address interliv address interliv memmori modul interliv thank advanc info robert                    |                 |
    | 34091 | comp_elec   | mb fd want subject say pleas email soon skcgoh tartarusuwaeduau                                      |                 |
    | 34152 | comp_elec   | perfect mag mxf monitor articl   last newsgroup                                                      |                 |
    | 34196 | comp_elec   | cga cardmonitor want titl say whi lee lee tosspotsvcom                                               |                 |
    | 34203 | comp_elec   | fastmicro busi heard fastmicro went busi thi true nt answer  number --                               |                 |
    | 34250 | comp_elec   | umbdrzip ani later version                                                                           |                 |
    | 34274 | comp_elec   | eric bosco eric send email address lost reconsid kevin                                               |                 |
    | 34325 | comp_elec   | diamond stealth  give  winmark dx stealth  vlb get  winmark ver                                      |                 |
    | 34330 | comp_elec   | test nt read -- gosh think instal viru wa call ms dos nt copi floppi burn love window crash          |                 |
    | 34336 | comp_elec   | smc e arcnet car sale ea  new smc e arcnet card sale brand new  wow cupportalcom walli waggon        |                 |
    | 34337 | comp_elec   | hay jt fax card sale  like new hay jt fax sale  offer trade walli waggon wow cupportalcom            |                 |
    | 34450 | comp_elec   | wincim  baud similar problem download use wincim discov disabl data compress modem work fine         |                 |
    | 34471 | comp_elec   | diamond stealth help articl   last newsgroup                                                         |                 |
    | 34473 | comp_elec   | hot cpu articl   last newsgroup                                                                      |                 |
    | 34483 | comp_elec   | need videotap pc output need videotap copi pc pd program pleas let know thi marc dna ucsusledu       |                 |
    | 34487 | comp_elec   | monitor - nanao hello follow discuss  monitor                                                        |                 |
    | 34548 | comp_elec   | monitor - nanao johnn eskimocom john navitski write hello follow discuss  monitor                    |                 |
    | 34658 | comp_elec   | help led connector motherboard articl   last newsgroup                                               |                 |
    | 34659 | comp_elec   | help ide drive instal problem articl   last newsgroup                                                |                 |
    | 34664 | comp_elec   | consum warn midwest micro ohio midwest micro articl   last newsgroup compdcommodem                   |                 |
    | 34667 | comp_elec   | dx vs dx articl   last newsgroup                                                                     |                 |
    | 34680 | comp_elec   | isa ca nt use  meg ram ok comput liter done ram  meg isa machin pleas e-mail thank advanc charley    |                 |
    | 34741 | comp_elec   | monitor - nanao in-reply-to johnn eskimocom messag  apr   gmt newsgroup                              |                 |
    | 34760 | comp_elec   | file server mac saw onc articl new line mac configur work optim file server anyon know ani detail    |                 |
    | 34771 | comp_elec   | wyse  termin emul wyse  termin emul comm toolbox kit avail net somewher thank vinc                   |                 |
    | 34863 | comp_elec   | sale powerbook slow articl martin tohi                                                               |                 |
    | 34872 | comp_elec   | lc vs rc centri  also use photoshop edit photo dtp work -nate                                        |                 |
    | 34907 | comp_elec   | modem question art                                                                                   |                 |
    | 34926 | comp_elec   | dead mous                                                                                            |                 |
    | 34927 | comp_elec   | x video iici need abl run nec fgx x  mode iici done right video card video card -michael             |                 |
    | 34939 | comp_elec   | jump start mac ii rememb clamp ground engin block first rob                                          |                 |
    | 34964 | comp_elec   | current rom version ship syquest drive titl say need know   c rom version steve -                    |                 |
    | 35012 | comp_elec   | get  x   monitor tri maxapplezoom sharewar init monitor driven intern video chen                     |                 |
    | 35040 | comp_elec   | mac suck buy pc thi test -- erm                                                                      |                 |
    | 35052 | comp_elec   | duo  crash aftersleep look like appl bug thi test                                                    |                 |
    | 35054 | comp_elec   | nt repair sticki mous button -- call appl -- -- -- x-post                                            |                 |
    | 35064 | comp_elec   | appl  monitor thi must faq veri first day  rgb better monitor well nec fgfgx pretti nice             |                 |
    | 35102 | comp_elec   | centri  math coprocessor option sorri thi faq nt normal read                                         |                 |
    | 35144 | comp_elec   | mac suck buy pc test suck post real messag                                                           |                 |
    | 35173 | comp_elec   | acceler classic ii doe one exist make much thank --                                                  |                 |
    | 35202 | comp_elec   | centri  math coprocessor option davidanthonyguevara cupportalcom write sorri thi faq nt normal read  |                 |
    | 35209 | comp_elec   | cach card iisi hi bought ago cach card w fpu techwork wa  think wa cheapest ever saw peter           |                 |
    | 35218 | comp_elec   | ntsc th                                                                                              |                 |
    | 35339 | comp_elec   | buy high speed veveryth modem hardwar handshak want use dan                                          |                 |
    | 35368 | comp_elec   | thi swim iwm use live  se mark - rom go -  thi solut person want upgrad fdhd cheer mike              |                 |
    | 35373 | comp_elec   | system l -- ernest stalnak jc sageccpurdueedu oo -- oo pur-e sagecc jc                               |                 |
    | 35415 | comp_elec   | number appli engin anyon phone number appli engin give call steven langloi slang bnrca               |                 |
    | 35418 | comp_elec   | fast modem slow mac                                                                                  |                 |
    | 35419 | comp_elec   | want audiomedia card want digidesgn audiomedia card mac email one sale thank eefc                    |                 |
    | 35426 | comp_elec   | c ugrad tempest go possibl upgrad c tempest motherboard switch probabl gon na expens right dt        |                 |
    | 35453 | comp_elec   | want macweek call macus magazin number guess give info                                               |                 |
    | 35528 | comp_elec   | macintosh weeni suck scsi disk think subject titl say anybodi reli scsi dick stoarag pain ass        |                 |
    | 35545 | comp_elec   | svga monitor centri real stori                                                                       |                 |
    | 35584 | comp_elec   | want adb mous keybd want appl adb mous keyboard contact paul gribbl abov email address asap paul g   |                 |
    | 35604 | comp_elec   | first tech ha anyon dealt first tech base austin tx ha experinc thank jame                           |                 |
    | 35635 | comp_elec   | monitor - kept  hour day articl apr allegedu                                                         |                 |
    | 35721 | comp_elec   | file server mac saw onc articl new line mac configur work optim file server anyon know ani detail    |                 |
    | 35732 | comp_elec   | wyse  termin emul wyse  termin emul comm toolbox kit avail net somewher thank vinc                   |                 |
    | 35824 | comp_elec   | sale powerbook slow articl martin tohi                                                               |                 |
    | 35833 | comp_elec   | lc vs rc centri  also use photoshop edit photo dtp work -nate                                        |                 |
    | 35868 | comp_elec   | modem question art                                                                                   |                 |
    | 35887 | comp_elec   | dead mous                                                                                            |                 |
    | 35888 | comp_elec   | x video iici need abl run nec fgx x  mode iici done right video card video card -michael             |                 |
    | 35900 | comp_elec   | jump start mac ii rememb clamp ground engin block first rob                                          |                 |
    | 35925 | comp_elec   | current rom version ship syquest drive titl say need know   c rom version steve -                    |                 |
    | 35973 | comp_elec   | get  x   monitor tri maxapplezoom sharewar init monitor driven intern video chen                     |                 |
    | 36001 | comp_elec   | mac suck buy pc thi test -- erm                                                                      |                 |
    | 36013 | comp_elec   | duo  crash aftersleep look like appl bug thi test                                                    |                 |
    | 36015 | comp_elec   | nt repair sticki mous button -- call appl -- -- -- x-post                                            |                 |
    | 36025 | comp_elec   | appl  monitor thi must faq veri first day  rgb better monitor well nec fgfgx pretti nice             |                 |
    | 36063 | comp_elec   | centri  math coprocessor option sorri thi faq nt normal read                                         |                 |
    | 36105 | comp_elec   | mac suck buy pc test suck post real messag                                                           |                 |
    | 36134 | comp_elec   | acceler classic ii doe one exist make much thank --                                                  |                 |
    | 36170 | comp_elec   | cach card iisi hi bought ago cach card w fpu techwork wa  think wa cheapest ever saw peter           |                 |
    | 36179 | comp_elec   | ntsc th                                                                                              |                 |
    | 36300 | comp_elec   | buy high speed veveryth modem hardwar handshak want use dan                                          |                 |
    | 36329 | comp_elec   | thi swim iwm use live  se mark - rom go -  thi solut person want upgrad fdhd cheer mike              |                 |
    | 36334 | comp_elec   | system l -- ernest stalnak jc sageccpurdueedu oo -- oo pur-e sagecc jc                               |                 |
    | 36376 | comp_elec   | number appli engin anyon phone number appli engin give call steven langloi slang bnrca               |                 |
    | 36379 | comp_elec   | fast modem slow mac                                                                                  |                 |
    | 36380 | comp_elec   | want audiomedia card want digidesgn audiomedia card mac email one sale thank eefc                    |                 |
    | 36387 | comp_elec   | c ugrad tempest go possibl upgrad c tempest motherboard switch probabl gon na expens right dt        |                 |
    | 36414 | comp_elec   | want macweek call macus magazin number guess give info                                               |                 |
    | 36489 | comp_elec   | macintosh weeni suck scsi disk think subject titl say anybodi reli scsi dick stoarag pain ass        |                 |
    | 36506 | comp_elec   | svga monitor centri real stori                                                                       |                 |
    | 36545 | comp_elec   | want adb mous keybd want appl adb mous keyboard contact paul gribbl abov email address asap paul g   |                 |
    | 36565 | comp_elec   | first tech ha anyon dealt first tech base austin tx ha experinc thank jame                           |                 |
    | 36596 | comp_elec   | monitor - kept  hour day articl apr allegedu                                                         |                 |
    | 36683 | comp_elec   | honor degre mean anyth ha thi got                                                                    |                 |
    | 36713 | comp_elec   | rotat text hi program xview suno  openwindow  would like rotat text display read faq                 |                 |
    | 36717 | comp_elec   | join x consortium hi doe anyon ani inform join x consortium cost benefit contact thank               |                 |
    | 36718 | comp_elec   | honor degre mean anyth justin kibel jck cattcitrieduau wrote ha thi got                              |                 |
    | 36775 | comp_elec   | expos event pleas excus previou post wa append thi thread accid -- robert                            |                 |
    | 36790 | comp_elec   | need xman sourc get xman sourc would rather get xman hp  sourc                                       |                 |
    | 36817 | comp_elec   | pleas ignor ideal oper system wa death blow unix whoop wrong group soooooooooooooooorri folk         |                 |
    | 36830 | comp_elec   | search xgolf xgolf program wa april fool joke sigh steve hite shite sinkholeunfedu                   |                 |
    | 36893 | comp_elec   | monthli question xcopyarea expos event excerpt netnew                                                |                 |
    | 36901 | comp_elec   | ask ftp address kerbero version  draft rfc                                                           |                 |
    | 36962 | comp_elec   | dell  eisa video card doe xfree support ani eisa video card dell  -- larri snyder larri gatorrncom   |                 |
    | 36980 | comp_elec   | subscrib subscrib grape nswsesnavymil                                                                |                 |
    | 37029 | comp_elec   | anim think tri send messag anim queri post                                                           |                 |
    | 37058 | comp_elec   | xterm statu line color look version xterm handl color vt style statu line anyon help thank           |                 |
    | 37066 | comp_elec   | dell  eisa video card thi belong                                                                     |                 |
    | 37103 | comp_elec   | athena widget find athena widget need xtdm- thank advanc                                             |                 |
    | 37104 | comp_elec   | subscrib subscrib min stellaskkuackr                                                                 |                 |
    | 37123 | comp_elec   | x toolkit excerpt netnew                                                                             |                 |
    | 37169 | comp_elec   | faq                                                                                                  |                 |
    | 37178 | comp_elec   | question found faq                                                                                   |                 |
    | 37184 | comp_elec   | look r xserver hp updat  hp-ux  get r server librari greg hugh gmh fchpcom                           |                 |
    | 37189 | comp_elec   | andrew wa x toolkit excerpt netnew                                                                   |                 |
    | 37193 | comp_elec   | x server nt doe anybodi x server nt share file experi bill steer westinghous                         |                 |
    | 37200 | comp_elec   | archie-cli xgetftp- need archi client program doe anybodi know get thank advanc kai                  |                 |
    | 37209 | comp_elec   | xarchie- avail export pleas accept follow announc                                                    |                 |
    | 37251 | comp_elec   | connect digitis x repost hi post thi                                                                 |                 |
    | 37288 | comp_elec   | motif server ascii termin doe anyon know x server charact cell termin doe nt anyth fanci long work   |                 |
    | 37296 | comp_elec   | subscrib pleas subscrib e-mail rpica portoinescnpt                                                   |                 |
    | 37298 | comp_elec   | subscriv pleas subscrib e-mail rpica portoinescnpt                                                   |                 |
    | 37306 | comp_elec   | get openbug motif  hello titl say need list bug motif  thank neal                                    |                 |
    | 37311 | comp_elec   | xr support i gx doe xr support graphic acceler board sun i thank advanc colin                        |                 |
    | 37321 | comp_elec   | subscrib pleas subscrib e-mail rpica portoinescnpt                                                   |                 |
    | 37350 | comp_elec   | x server scanlin pad question figur answer lie mitserverddxmfbmfbcustomh brian                       |                 |
    | 37407 | comp_elec   | none ubject subscriv pleas subscrib e-mail min stellaskkuackr                                                                                                      |                 |
    | 37420 | comp_elec   | tn support xterm tn program support xterm nt like x ca nt copi window thank                          |                 |
    | 37514 | comp_elec   | drop                                                                                                 |                 |
    | 37534 | comp_elec   | x termin faq  current state world x termin jim morton jim applixcom post quarterli                   |                 |
    | 37561 | comp_elec   | none subscrib xpert skji evekaistackr                                                                |                 |
    | 37562 | comp_elec   | faq could someon repost faq thi group pleas thank ladisla                                            |                 |
    | 37563 | comp_elec   | unsubscrib unsubscrib                                                                                |                 |
    | 37564 | comp_elec   | color x window aicon excerpt netnew                                                                  |                 |
    | 37567 | comp_elec   | unsubscrib unsubscrib                                                                                |                 |
    | 37613 | comp_elec   | hpgl anyth convert want                                                                              |                 |
    | 37646 | comp_elec   |                                                                                                      |                 |
    | 37654 | comp_elec   | honor degre mean anyth ha thi got                                                                    |                 |
    | 37670 | comp_elec   |                                                                                                      |                 |
    | 37672 | comp_elec   |                                                                                                      |                 |
    | 37677 | comp_elec   |                                                                                                      |                 |
    | 37678 | comp_elec   |                                                                                                      |                 |
    | 37681 | comp_elec   |                                                                                                      |                 |
    | 37689 | comp_elec   | rotat text hi program xview suno  openwindow  would like rotat text display read faq                 |                 |
    | 37693 | comp_elec   | join x consortium hi doe anyon ani inform join x consortium cost benefit contact thank               |                 |
    | 37694 | comp_elec   | honor degre mean anyth justin kibel jck cattcitrieduau wrote ha thi got                              |                 |
    | 37718 | comp_elec   |                                                                                                      |                 |
    | 37752 | comp_elec   | expos event pleas excus previou post wa append thi thread accid -- robert                            |                 |
    | 37767 | comp_elec   | need xman sourc get xman sourc would rather get xman hp  sourc                                       |                 |
    | 37794 | comp_elec   | pleas ignor ideal oper system wa death blow unix whoop wrong group soooooooooooooooorri folk         |                 |
    | 37807 | comp_elec   | search xgolf xgolf program wa april fool joke sigh steve hite shite sinkholeunfedu                   |                 |
    | 37869 | comp_elec   | monthli question xcopyarea expos event excerpt netnew                                                |                 |
    | 37877 | comp_elec   | ask ftp address kerbero version  draft rfc                                                           |                 |
    | 37917 | comp_elec   |                                                                                                      |                 |
    | 37939 | comp_elec   | dell  eisa video card doe xfree support ani eisa video card dell  -- larri snyder larri gatorrncom   |                 |
    | 37957 | comp_elec   | subscrib subscrib grape nswsesnavymil                                                                |                 |
    | 38006 | comp_elec   | anim think tri send messag anim queri post                                                           |                 |
    | 38035 | comp_elec   | xterm statu line color look version xterm handl color vt style statu line anyon help thank           |                 |
    | 38043 | comp_elec   | dell  eisa video card thi belong                                                                     |                 |
    | 38080 | comp_elec   | athena widget find athena widget need xtdm- thank advanc                                             |                 |
    | 38081 | comp_elec   | subscrib subscrib min stellaskkuackr                                                                 |                 |
    | 38100 | comp_elec   | x toolkit excerpt netnew                                                                             |                 |
    | 38147 | comp_elec   | faq                                                                                                  |                 |
    | 38156 | comp_elec   | question found faq                                                                                   |                 |
    | 38162 | comp_elec   | look r xserver hp updat  hp-ux  get r server librari greg hugh gmh fchpcom                           |                 |
    | 38167 | comp_elec   | andrew wa x toolkit excerpt netnew                                                                   |                 |
    | 38171 | comp_elec   | x server nt doe anybodi x server nt share file experi bill steer westinghous                         |                 |
    | 38178 | comp_elec   | archie-cli xgetftp- need archi client program doe anybodi know get thank advanc kai                  |                 |
    | 38187 | comp_elec   | xarchie- avail export pleas accept follow announc                                                    |                 |
    | 38199 | comp_elec   |                                                                                                      |                 |
    | 38230 | comp_elec   | connect digitis x repost hi post thi                                                                 |                 |
    | 38267 | comp_elec   | motif server ascii termin doe anyon know x server charact cell termin doe nt anyth fanci long work   |                 |
    | 38275 | comp_elec   | subscrib pleas subscrib e-mail rpica portoinescnpt                                                   |                 |
    | 38277 | comp_elec   | subscriv pleas subscrib e-mail rpica portoinescnpt                                                   |                 |
    | 38285 | comp_elec   | get openbug motif  hello titl say need list bug motif  thank neal                                    |                 |
    | 38290 | comp_elec   | xr support i gx doe xr support graphic acceler board sun i thank advanc colin                        |                 |
    | 38300 | comp_elec   | subscrib pleas subscrib e-mail rpica portoinescnpt                                                   |                 |
    | 38329 | comp_elec   | x server scanlin pad question figur answer lie mitserverddxmfbmfbcustomh brian                       |                 |
    | 38386 | comp_elec   | none ubject subscriv pleas subscrib e-mail min stellaskkuackr                                                                                                      |                 |
    | 38399 | comp_elec   | tn support xterm tn program support xterm nt like x ca nt copi window thank                          |                 |
    | 38407 | comp_elec   |                                                                                                      |                 |
    | 38494 | comp_elec   | drop                                                                                                 |                 |
    | 38514 | comp_elec   | x termin faq  current state world x termin jim morton jim applixcom post quarterli                   |                 |
    | 38541 | comp_elec   | none subscrib xpert skji evekaistackr                                                                |                 |
    | 38542 | comp_elec   | faq could someon repost faq thi group pleas thank ladisla                                            |                 |
    | 38543 | comp_elec   | unsubscrib unsubscrib                                                                                |                 |
    | 38544 | comp_elec   | color x window aicon excerpt netnew                                                                  |                 |
    | 38547 | comp_elec   | unsubscrib unsubscrib                                                                                |                 |
    | 38593 | comp_elec   | hpgl anyth convert want                                                                              |                 |
    | 38646 | comp_elec   | principleofthebreathalyz nz appar thing like aftershav also give posit read                          |                 |
    | 38677 | comp_elec   | homebuilt pal epld program                                                                           |                 |
    | 38685 | comp_elec   | booksinfo audio dsp                                                                                  |                 |
    | 38690 | comp_elec   | self-modifi hardwar permit quot fragment praetzel suneeuwaterlooca articl context -newsgroup         |                 |
    | 38693 | comp_elec   | radar detector detector detect oscil oper detector saw stori use canada nt go put oscil car -        |                 |
    | 38704 | comp_elec   | video io idea anyon idea build cheap low resolut high - video projector exampl lcd slide projector   |                 |
    | 38736 | comp_elec   | electr power line ball know airport nearbi may marker tell pilot small plane power line nearbi joe   |                 |
    | 38806 | comp_elec   | electr wire faq wa question vac outlet wire sinc electr wire question turn time time                 |                 |
    | 38882 | comp_elec   | microwav use collect xyz coordin get info brochur differenti gp system buy bobc                      |                 |
    | 38898 | comp_elec   | nuclear site cool tower excerpt netnew                                                               |                 |
    | 38907 | comp_elec   | nuclear site cool tower wayn alan martin wmh andrewcmuedu write excerpt netnew                       |                 |
    | 38979 | comp_elec   | bit serial convert someon wa look week ago - check compdsp mike                                      |                 |
    | 38993 | comp_elec   | doe someon know news group ieee yxy usledu thank lot                                                 |                 |
    | 38999 | comp_elec   | dayton hamfest ye    doe anyon direct get get dayton thank wayn martin                               |                 |
    | 39017 | comp_elec   | circuit cellar ink address cci still publish doe anyon address                                       |                 |
    | 39099 | comp_elec   | hc public domain softwar doe nt motorola amcu someth bb yet --                                       |                 |
    | 39166 | comp_elec   | lo angel freeway traffic report oop knx  knbr frisco  doug san fran ca nt citi jack webb told claar  |                 |
    | 39176 | comp_elec   | pink nois pink nois use sound experi -toni wayn uvaschoolsvirginiaedu                                |                 |
    | 39179 | comp_elec   | part mcsece know hce tri figur sece specif doe sec stand -- -- dale ulan vedau ulan eeualbertaca     |                 |
    | 39188 | comp_elec   | anyon got  schadow switch led name price cap colour quantiti avail cheer mike                        |                 |
    | 39200 | comp_elec   | need sourc old radio shack ste made rohm baxxx part call -- ask get sampl onli like  part            |                 |
    | 39210 | comp_elec   | pink nois                                                                                            |                 |
    | 39212 | comp_elec   | need info dsp project file simtel archiv call addazip think dsp                                      |                 |
    | 39216 | comp_elec   | radio freq use measur distanc                                                                        |                 |
    | 39229 | comp_elec   | make odd resistor valu requir filter                                                                 |                 |
    | 39242 | comp_elec   | want scope look  mhz scope good condit pleas email call  -                                           |                 |
    | 39250 | comp_elec   | radio freq use measur distanc articl  otterhplhpcom tgg otterhplhpcom tom gardner write              |                 |
    | 39251 | comp_elec   | test thi test see thi work                                                                           |                 |
    | 39294 | comp_elec   | pcmcia                                                                                               |                 |
    | 39297 | comp_elec   | electron design mag                                                                                  |                 |
    | 39307 | comp_elec   | microstep doe anyon know get schemat micro step circuit ani help would appreci mcole nmsuedu         |                 |
    | 39308 | comp_elec   | hm hm mice chip number hm hm abl find inform ani help would appreci mcole nmsuedu                    |                 |
    | 39346 | comp_elec   | need help find part ok post thi b blue waveqwk v                                                     |                 |
    | 39350 | comp_elec   | food deydrat articl                                                                                  |                 |
    | 39376 | comp_elec   | doe ani one know biggest rom present pleas replay yxy usledu thank lot                               |                 |
    | 39514 | comp_elec   | doe ani one know biggest rom present pleas replay yxy usledu thank lot                               |                 |
    | 39545 | comp_elec   | card phone help understand cardphon oper valu store phonecard thanx                                  |                 |
    | 39550 | comp_elec   | hd-tv sound system would like get inform current system use hd-tv sound systemsthank                 |                 |
    | 39567 | comp_elec   | whi circuit board green                                                                              |                 |
    | 39631 | comp_elec   | principleofthebreathalyz nz appar thing like aftershav also give posit read                          |                 |
    | 39662 | comp_elec   | homebuilt pal epld program                                                                           |                 |
    | 39670 | comp_elec   | booksinfo audio dsp                                                                                  |                 |
    | 39675 | comp_elec   | self-modifi hardwar permit quot fragment praetzel suneeuwaterlooca articl context -newsgroup         |                 |
    | 39678 | comp_elec   | radar detector detector detect oscil oper detector saw stori use canada nt go put oscil car -        |                 |
    | 39689 | comp_elec   | video io idea anyon idea build cheap low resolut high - video projector exampl lcd slide projector   |                 |
    | 39721 | comp_elec   | electr power line ball know airport nearbi may marker tell pilot small plane power line nearbi joe   |                 |
    | 39791 | comp_elec   | electr wire faq wa question vac outlet wire sinc electr wire question turn time time                 |                 |
    | 39867 | comp_elec   | microwav use collect xyz coordin get info brochur differenti gp system buy bobc                      |                 |
    | 39883 | comp_elec   | nuclear site cool tower excerpt netnew                                                               |                 |
    | 39892 | comp_elec   | nuclear site cool tower wayn alan martin wmh andrewcmuedu write excerpt netnew                       |                 |
    | 39964 | comp_elec   | bit serial convert someon wa look week ago - check compdsp mike                                      |                 |
    | 39978 | comp_elec   | doe someon know news group ieee yxy usledu thank lot                                                 |                 |
    | 39984 | comp_elec   | dayton hamfest ye    doe anyon direct get get dayton thank wayn martin                               |                 |
    | 40002 | comp_elec   | circuit cellar ink address cci still publish doe anyon address                                       |                 |
    | 40084 | comp_elec   | hc public domain softwar doe nt motorola amcu someth bb yet --                                       |                 |
    | 40161 | comp_elec   | pink nois pink nois use sound experi -toni wayn uvaschoolsvirginiaedu                                |                 |
    | 40164 | comp_elec   | part mcsece know hce tri figur sece specif doe sec stand -- -- dale ulan vedau ulan eeualbertaca     |                 |
    | 40173 | comp_elec   | anyon got  schadow switch led name price cap colour quantiti avail cheer mike                        |                 |
    | 40185 | comp_elec   | need sourc old radio shack ste made rohm baxxx part call -- ask get sampl onli like  part            |                 |
    | 40195 | comp_elec   | pink nois                                                                                            |                 |
    | 40197 | comp_elec   | need info dsp project file simtel archiv call addazip think dsp                                      |                 |
    | 40201 | comp_elec   | radio freq use measur distanc                                                                        |                 |
    | 40214 | comp_elec   | make odd resistor valu requir filter                                                                 |                 |
    | 40227 | comp_elec   | want scope look  mhz scope good condit pleas email call  -                                           |                 |
    | 40235 | comp_elec   | radio freq use measur distanc articl  otterhplhpcom tgg otterhplhpcom tom gardner write              |                 |
    | 40236 | comp_elec   | test thi test see thi work                                                                           |                 |
    | 40279 | comp_elec   | pcmcia                                                                                               |                 |
    | 40280 | comp_elec   | electron design mag                                                                                  |                 |
    | 40288 | comp_elec   | microstep doe anyon know get schemat micro step circuit ani help would appreci mcole nmsuedu         |                 |
    | 40289 | comp_elec   | hm hm mice chip number hm hm abl find inform ani help would appreci mcole nmsuedu                    |                 |
    | 40327 | comp_elec   | need help find part ok post thi b blue waveqwk v                                                     |                 |
    | 40331 | comp_elec   | food deydrat articl                                                                                  |                 |
    | 40357 | comp_elec   | doe ani one know biggest rom present pleas replay yxy usledu thank lot                               |                 |
    | 40495 | comp_elec   | doe ani one know biggest rom present pleas replay yxy usledu thank lot                               |                 |
    | 40526 | comp_elec   | card phone help understand cardphon oper valu store phonecard thanx                                  |                 |
    | 40531 | comp_elec   | hd-tv sound system would like get inform current system use hd-tv sound systemsthank                 |                 |
    | 40548 | comp_elec   | whi circuit board green                                                                              |                 |



```python
df.iloc[39]
```




    newsgroup                                                  sci_med
    body             mysteri ill eye problem friend ha follow sympt...
    exploded_body    mysteri ill eye problem friend ha follow sympt...
    Name: 27, dtype: object




```python
df.iloc[40]
```




    newsgroup                                                  sci_med
    body             mysteri ill eye problem friend ha follow sympt...
    exploded_body    acut quit concern becaus retin hemorrhag becom...
    Name: 27, dtype: object




```python
df.iloc[42]
```




    newsgroup                                                  sci_med
    body             new diet -- work great articl apr inmetcambinm...
    exploded_body    recent version adipos problem anecdot report i...
    Name: 28, dtype: object




```python
bad_indices = df[df['exploded_body'].apply(lambda x: not isinstance(x, str))].index
```


```python
df.drop(bad_indices, inplace = True)
```


```python
df.shape
```




    (69177, 3)




```python
df.to_pickle('../data/dataframes/newsgroup_body_cleaned_exploded.pkl')
```


```python
all_categories = ['sport', 'autos', 'religion', 'comp_elec', 'sci_med', 'seller', 'politics']
# We'll use all
target_categories = ['sport', 'autos', 'religion', 'comp_elec', 'sci_med', 'seller', 'politics']
```


```python
# container for sentences
X = np.array([t for t in df['exploded_body']])
# container for sentences
y = np.array([n for n in df['newsgroup']])
```


```python
encoder = LabelEncoder()
y = encoder.fit_transform(df['newsgroup'])
```


```python
y.shape
```




    (69177,)




```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y, 
                                                    test_size=0.25)

classes = np.unique(y_train)
mapping = dict(zip(classes, target_categories))

len(X_train), len(X_test), classes, mapping
```




    (51882,
     17295,
     array([0, 1, 2, 3, 4, 5, 6]),
     {0: 'sport',
      1: 'autos',
      2: 'religion',
      3: 'comp_elec',
      4: 'sci_med',
      5: 'seller',
      6: 'politics'})




```python
# model parameters
vocab_size = 1200
embedding_dim = 16
max_length = 120
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
```


```python
# tokenize sentences
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index

# convert train dataset to sequence and pad sequences
train_sequences = tokenizer.texts_to_sequences(X_train)
train_padded = pad_sequences(train_sequences, padding=padding_type, maxlen=max_length)

# convert validation dataset to sequence and pad sequences
validation_sequences = tokenizer.texts_to_sequences(X_test)
validation_padded = pad_sequences(validation_sequences, padding=padding_type, maxlen=max_length)
```


```python
### Final layer must be same as y.shape output -> # categories
```


```python
# model initialization
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(7, activation='sigmoid')
])
model._name = "NewsgroupBodyCleanBOW"
# compile model
# categorical-cross-entropy requires labels one-hot-encoded. sparse = as ints. binary = t/f
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# model summary
print(model.summary())
```

    Model: "NewsgroupBodyCleanBOW"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     embedding_1 (Embedding)     (None, 120, 16)           19200     
                                                                     
     global_average_pooling1d_1   (None, 16)               0         
     (GlobalAveragePooling1D)                                        
                                                                     
     dense_2 (Dense)             (None, 24)                408       
                                                                     
     dense_3 (Dense)             (None, 7)                 175       
                                                                     
    =================================================================
    Total params: 19,783
    Trainable params: 19,783
    Non-trainable params: 0
    _________________________________________________________________
    None



```python
# fit model
num_epochs = 20
history = model.fit(train_padded, y_train, 
                    epochs=num_epochs, verbose=1,
                    validation_split=0.3)

# predict values
pred = model.predict(validation_padded)
```

    Epoch 1/20
    1135/1135 [==============================] - 5s 3ms/step - loss: 1.3570 - accuracy: 0.4956 - val_loss: 0.9589 - val_accuracy: 0.6743
    Epoch 2/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.7806 - accuracy: 0.7425 - val_loss: 0.6945 - val_accuracy: 0.7727
    Epoch 3/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.6152 - accuracy: 0.8015 - val_loss: 0.5928 - val_accuracy: 0.8094
    Epoch 4/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.5274 - accuracy: 0.8303 - val_loss: 0.5361 - val_accuracy: 0.8285
    Epoch 5/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.4730 - accuracy: 0.8470 - val_loss: 0.4974 - val_accuracy: 0.8406
    Epoch 6/20
    1135/1135 [==============================] - 4s 4ms/step - loss: 0.4379 - accuracy: 0.8571 - val_loss: 0.4825 - val_accuracy: 0.8468
    Epoch 7/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.4147 - accuracy: 0.8647 - val_loss: 0.4677 - val_accuracy: 0.8490
    Epoch 8/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.3959 - accuracy: 0.8697 - val_loss: 0.4553 - val_accuracy: 0.8510
    Epoch 9/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.3811 - accuracy: 0.8736 - val_loss: 0.4470 - val_accuracy: 0.8536
    Epoch 10/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.3691 - accuracy: 0.8787 - val_loss: 0.4403 - val_accuracy: 0.8563
    Epoch 11/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.3581 - accuracy: 0.8806 - val_loss: 0.4384 - val_accuracy: 0.8545
    Epoch 12/20
    1135/1135 [==============================] - 4s 4ms/step - loss: 0.3484 - accuracy: 0.8850 - val_loss: 0.4363 - val_accuracy: 0.8566
    Epoch 13/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.3399 - accuracy: 0.8865 - val_loss: 0.4303 - val_accuracy: 0.8583
    Epoch 14/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.3325 - accuracy: 0.8889 - val_loss: 0.4267 - val_accuracy: 0.8605
    Epoch 15/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.3253 - accuracy: 0.8917 - val_loss: 0.4265 - val_accuracy: 0.8649
    Epoch 16/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.3197 - accuracy: 0.8935 - val_loss: 0.4240 - val_accuracy: 0.8630
    Epoch 17/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.3133 - accuracy: 0.8960 - val_loss: 0.4225 - val_accuracy: 0.8639
    Epoch 18/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.3081 - accuracy: 0.8981 - val_loss: 0.4207 - val_accuracy: 0.8659
    Epoch 19/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.3029 - accuracy: 0.9004 - val_loss: 0.4205 - val_accuracy: 0.8666
    Epoch 20/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.2981 - accuracy: 0.9015 - val_loss: 0.4228 - val_accuracy: 0.8671
    541/541 [==============================] - 1s 2ms/step



```python
import os

file_name = 'run_01'
plot_type = 'history'
model_name = 'newsgroup_body_clean'
#####
os.makedirs(f"images/{plot_type}", exist_ok=True)
os.makedirs(f"images/{plot_type}/{model_name}", exist_ok=True)
save_path = f'images/{plot_type}/{model_name}/{file_name}.png' 

utils.plot_history_and_save(history, save_path)
```


![png](clean_run_01_files/clean_run_01_30_0.png)



```python
# TensorFlow SavedModel format => .keras
model_file = 'models/newsgroup_body_clean_model'
model.save(model_file)
```

    WARNING:absl:Found untraced functions such as _update_step_xla while saving (showing 1 of 1). These functions will not be directly callable after loading.


    INFO:tensorflow:Assets written to: models/newsgroup_body_clean_model/assets


    INFO:tensorflow:Assets written to: models/newsgroup_body_clean_model/assets


That just warms me up a bit. Now, lets take a look at what some augmentation can do. I am not expecting too much of a change here to be honest, and am actually curious if we will experience any setbacks and what some of the initial runs are like in the epochs.


```python

```
