# Wikipedia corpus-v3 review packet

This is the human validation gate between building the Wikipedia-specific quality policy and adapting quality logic to the other six data lanes.

## What to learn from this stage

The signals are measurements, the band is a policy decision derived from those measurements, the weight controls ordinary sampling, and a cap is a final safety ceiling. A cap may be configured without activating.

## Population and weighted supply

- Physical records: 5,183
- Weighted records before caps: 5,164.90
- Review examples: 62

| Band | Physical records |
|---|---:|
| B0 | 820 |
| B1 | 1 |
| B2 | 1,328 |
| B3 | 1,611 |
| B4 | 1,423 |

## Do the caps currently activate?

| Group | Records | Share after weights | Cap | Activates? |
|---|---:|---:|---:|---|
| general_short | 304 | 1.47% | 1.00% | yes |
| general_disambiguation | 106 | 0.51% | 2.00% | no |
| general_structured_low_prose | 26 | 0.13% | 2.00% | no |
| general_linewise_list | 336 | 1.63% | 3.00% | no |
| general_category_tail | 27 | 0.13% | 0.50% | no |
| general_stat_heavy_list | 1 | 0.00% | 1.00% | no |
| general_table_salvage | 84 | 0.41% | 1.00% | no |
| general_sensitive_context_review | 1 | 0.00% | 0.10% | no |
| all_B0_combined | 820 | 3.97% | 5.00% | no |

Only an activating cap changes the distribution beyond the sampling weights. Non-activating caps remain useful as guards if the corpus grows later.

## Review rubric

For each example, inspect whether it contains meaningful language, is coherent and complete, belongs in its assigned band/cap group, is PII-safe, and ends cleanly. Then choose keep, downweight, or reject. Do not change a threshold after one unusual example; look for a repeated error pattern.

## Deterministic sample

Five examples are drawn across the length range of every band and cap group. All 9 non-paragraph boundary chunks are included. Some records intentionally appear in more than one stratum because they test different claims.

### band_B0

#### Caning (`rec_4e5c950ad8c84349677fea8a`)

Band/weight: B0 / 0.25; caps: ['general_category_tail']; flags: ['category_tail', 'short_continuation_chunk']

Chunk: 3/3; end boundary: paragraph; characters: 77; paragraphs: 2; alpha: 0.883; repeated trigrams: 0.000

Beginning:

> Caning BDSM activities Corporal punishments School punishments Whipping Pain

Ending:

> Caning BDSM activities Corporal punishments School punishments Whipping Pain

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Geech Yarborough (`rec_86e75c50eb631b0dbcb0be3a`)

Band/weight: B0 / 0.25; caps: ['general_short']; flags: ['short_document']

Chunk: 1/1; end boundary: paragraph; characters: 358; paragraphs: 5; alpha: 0.813; repeated trigrams: 0.019

Beginning:

> Geech Yarborough Geech Yarborough was an American baseball catcher and pitcher in the Negro leagues. He played with the Atlanta Black Crackers in 1932 and the Newark Eagles in 1940. References External links and Seamheads Atlanta Black Crackers players Newark Eagles players Year of birth missing Year of death missing Baseball pitchers Baseball catchers

Ending:

> Geech Yarborough Geech Yarborough was an American baseball catcher and pitcher in the Negro leagues. He played with the Atlanta Black Crackers in 1932 and the Newark Eagles in 1940. References External links and Seamheads Atlanta Black Crackers players Newark Eagles players Year of birth missing Year of death missing Baseball pitchers Baseball catchers

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### List of ambassadors of Guinea to the United States (`rec_3eb114f7aeb875dfed808120`)

Band/weight: B0 / 0.25; caps: ['general_linewise_list']; flags: ['linewise_list']

Chunk: 1/1; end boundary: paragraph; characters: 764; paragraphs: 5; alpha: 0.615; repeated trigrams: 0.050

Beginning:

> List of ambassadors of Guinea to the United States Below is the list of ambassadors from Guinea to the United States: Diallo Telli (1959–61) Conté Seydou (1961–68) Karim Bangoura (1969–71) Keita Mory (1971–72) Touré Sadam Moussa (1972–74) Bah Habib (1974–76) Kouroma Daouda (1977) Camara Ibrahima (1977–79) Condé Mohamed Laminé (1979–83) Diallo Thierno Habib (1983–84) Beavogui Tollo (1984–88) Camara Kékoura (1988–90) Sangaré Moussa (1990–93) Barry Boubacar (1993–96) Thiam Mohamed Aly (1996–2001) Barry Rafiou Alpha Oumar (2002-) Blaise chérif (2011 - 2014) Mamady Condé (2014-2017) Kerfalla Yansané (2017-2022 Fatoumata Kaba (2022- ) References United States of America, Ambassadors from Guinea t…

Ending:

> …he list of ambassadors from Guinea to the United States: Diallo Telli (1959–61) Conté Seydou (1961–68) Karim Bangoura (1969–71) Keita Mory (1971–72) Touré Sadam Moussa (1972–74) Bah Habib (1974–76) Kouroma Daouda (1977) Camara Ibrahima (1977–79) Condé Mohamed Laminé (1979–83) Diallo Thierno Habib (1983–84) Beavogui Tollo (1984–88) Camara Kékoura (1988–90) Sangaré Moussa (1990–93) Barry Boubacar (1993–96) Thiam Mohamed Aly (1996–2001) Barry Rafiou Alpha Oumar (2002-) Blaise chérif (2011 - 2014) Mamady Condé (2014-2017) Kerfalla Yansané (2017-2022 Fatoumata Kaba (2022- ) References United States of America, Ambassadors from Guinea to Guinea es:Anexo:Embajadores de Guinea en los Estados Unidos

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Grand Duke Nicholas Nikolaevich of Russia (1856–1929) (`rec_ca8aaedd70dc3cd7087fb1e8`)

Band/weight: B0 / 0.25; caps: ['general_linewise_list']; flags: ['linewise_list']

Chunk: 3/3; end boundary: paragraph; characters: 3,006; paragraphs: 10; alpha: 0.750; repeated trigrams: 0.202

Beginning:

> Grand Duke Nicholas Nikolaevich of Russia (1856–1929) Foreign Knight of the Order of the Most Holy Annunciation (Kingdom of Italy) – 18 June 1890 – during a visit to Russia of King Victor Emmanuel III Knight of the Order of the Elephant (Denmark) – 19 July 1909 Grand Cross of the Order of the Redeemer (Kingdom of Greece) Grand Cross of the Ludwig Order (Grand Duchy of Hesse and by Rhine) – 10 March 1886 Grand Cross of the House Order of the Wendish Crown (Mecklenburg) Grand Cross of the Order of Danilo I (Principality of Montenegro) Grand Cross of the House and Merit Order of Peter Frederick Louis, with Golden Crown (Grand Duchy of Oldenburg) – 7 December 1856 Knight of the Order of the Bla…

Ending:

> …iversity alumni Russian military personnel of World War I Military leaders of the Russian Empire Emigrants from the Russian Empire to Italy Recipients of the Order of St. George of the Second Degree Recipients of the Order of St. George of the Third Degree Grand Cross of the Legion of Honour Grand Crosses of the Order of Saint Stephen of Hungary 19th-century people from the Russian Empire Emigrants from the Russian Empire to France Anti-communists from the Russian Empire Monarchists from the Russian Empire Pretenders to the Russian throne Burials at Bratsky Cemetery, Moscow World War I crimes by the Russian Empire Military personnel from Saint Petersburg Russian mass murderers War criminals

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Khaliji (music) (`rec_b31f126054d03adf1e205d4b`)

Band/weight: B0 / 0.25; caps: ['general_linewise_list']; flags: ['linewise_list']

Chunk: 1/2; end boundary: paragraph; characters: 7,998; paragraphs: 44; alpha: 0.826; repeated trigrams: 0.015

Beginning:

> Khaliji (music) Khaliji music (also spelled Khaleeji; meaning Gulf music) is the music of Eastern Arabia, the Arab states of the Persian Gulf and it is popular across the Arab world. It is traditionally characterized by heavy use of the rebab, oud and other string instruments such as the violin, the occasional use of habbān, and the inclusion of percussion instruments such as the mirwas, tabl, and duff drums. Khaliji music first started as a bedouin tradition with poetry sung by a tribe's shaa'ir, which means poet, usually accompanied by a rebab, the lyrics dealt with tales of honor, love, camel riders, and glory warriors. Khaliji music has roots going back more than 1,000 years, to the Isl…

Ending:

> …i Cheb Faris El Sataifi Spain Hakim (Spanish singer) Greece Grigoris Asikis Konstantinos Argyros Yiorgos Batis George Dalaras Anestis Delias Stratos Pagioumtzis Giorgos Xylouris Babis Tsertos Mariza Koch United Kingdom Yusuf Islam Sami Yusuf Iran Evin Agassi Nematollah Aghasi Hooshmand Aghili Salar Aghili Morteza Ahmadi Alireza Assar Davood Azad Mohsen Chavoshi Farman Fathalian Farzad Fattahi Babak Jahanbakhsh Shahrum Kashani Ehsan Khajeh Amiri Ali Lohrasbi Morteza Pashaei Rahim Shahriari Reza Yazdani Mohsen Yeganeh Sima Bina Leila Forouhar Googoosh Mahasti Marjan (singer) Giti Pashaei Israel Etti Ankri Zohra Al Fassiya Yael Naim Cyprus Hovig Demirjian See also Arabic music Arabic pop music

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

### band_B1

#### SV Arminen (`rec_e0b755e0b8e7085261498ef7`)

Band/weight: B1 / 0.5; caps: none; flags: ['low_language_content']

Chunk: 1/1; end boundary: paragraph; characters: 1,780; paragraphs: 9; alpha: 0.446; repeated trigrams: 0.161

Beginning:

> SV Arminen Sportvereinigung Arminen Wien also known as SV Arminen or simply Arminen is an Austrian professional field hockey club based in Vienna. It competes in the Austrian Bundesliga which they have won a record 19 times in the men's competition. Their home ground is the Waldstadion and they were founded on 24 April 1919. The first men's team regularly plays in the Euro Hockey League. In the past, the club also had other sports but since 1938 they only have hockey. Honours Men Austrian Bundesliga Winners (20): 1945–46, 1946–47, 1947–48, 1948–49, 1950–51, 1952–53, 1976, 1980, 1983, 1985, 1986, 1987, 1988, 2012–13, 2013–14, 2015–16, 2016–17, 2017–18, 2018–19, 2021–22 EuroHockey Club Trophy…

Ending:

> …Winners (1): 2014 Runners-up (1): 2004 Women Austrian Bundesliga Winners (20): 1948–49, 1957–58, 1978, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1992, 1998–99, 2002–03, 2011–12, 2012–13, 2013–14, 2014–15, 2015–16, 2016–17, 2017–18 Austrian Indoor Bundesliga Winners (21): 1958–59, 1980, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1992, 1997–98, 2009–10, 2010–11, 2011–12, 2013–14, 2014–15, 2015–16, 2016–17, 2017–18, 2018–19, 2019–20 EuroHockey Indoor Club Trophy Winners (3): 2011, 2015, 2019 EuroHockey Cup Winners Trophy Winners (1): 1992 References External links Field hockey clubs in Austria Field hockey clubs established in 1919 1919 establishments in Austria Sports clubs and teams in Vienna

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

### band_B2

#### Siyavashabad-e Chendar (`rec_0c516f94a4338e7367bf7f4e`)

Band/weight: B2 / 1.0; caps: none; flags: none

Chunk: 1/1; end boundary: paragraph; characters: 400; paragraphs: 4; alpha: 0.795; repeated trigrams: 0.054

Beginning:

> Siyavashabad-e Chendar Siyavashabad-e Chendar (, also Romanized as Sīyāvashābād-e Chendār) is a village in Doab Rural District, Bazoft District, Kuhrang County, Chaharmahal and Bakhtiari Province, Iran. At the 2006 census, its population was 169, in 30 families. The village is populated by Lurs. References Populated places in Kuhrang County Luri settlements in Chaharmahal and Bakhtiari Province

Ending:

> Siyavashabad-e Chendar Siyavashabad-e Chendar (, also Romanized as Sīyāvashābād-e Chendār) is a village in Doab Rural District, Bazoft District, Kuhrang County, Chaharmahal and Bakhtiari Province, Iran. At the 2006 census, its population was 169, in 30 families. The village is populated by Lurs. References Populated places in Kuhrang County Luri settlements in Chaharmahal and Bakhtiari Province

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### 2011–12 Professional Hockey League season (`rec_a643d39aa6cd339d95652758`)

Band/weight: B2 / 1.0; caps: none; flags: none

Chunk: 1/1; end boundary: paragraph; characters: 537; paragraphs: 6; alpha: 0.788; repeated trigrams: 0.074

Beginning:

> 2011–12 Professional Hockey League season The 2011–12 Professional Hockey League season was the 20th annual edition of the Ukrainian Hockey Championship held in 2011–12. The season marked the first season of the Professional Hockey League and first time the national title was administered and awarded independently of the Ice Hockey Federation of Ukraine (FHU). Eight teams participated in the league, which was won by HC Donbass-2. Regular season Playoffs External links Official website Uk Professional Hockey League seasons Prof

Ending:

> 2011–12 Professional Hockey League season The 2011–12 Professional Hockey League season was the 20th annual edition of the Ukrainian Hockey Championship held in 2011–12. The season marked the first season of the Professional Hockey League and first time the national title was administered and awarded independently of the Ice Hockey Federation of Ukraine (FHU). Eight teams participated in the league, which was won by HC Donbass-2. Regular season Playoffs External links Official website Uk Professional Hockey League seasons Prof

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Mbereshi Girls' School (`rec_423db6352852a5adc9a9262c`)

Band/weight: B2 / 1.0; caps: none; flags: none

Chunk: 1/1; end boundary: paragraph; characters: 713; paragraphs: 6; alpha: 0.773; repeated trigrams: 0.030

Beginning:

> Mbereshi Girls' School Mbereshi Girls' School was a mission boarding school at Mbereshi. As "the earliest girls' school in Northern Rhodesia ... this school gained an international reputation." Mbereshi Girls' Boarding School was founded by the missionary Mabel Shaw in 1915, and Shaw served as its Principal until 1940. In 1946 the school was combined with the boy's boarding school to form a new coeducational institution. Alumni Betty Kaunda (1928-2012), First Lady of Zambia References Girls' schools in Zambia Educational institutions established in 1915 Boarding schools in Zambia 1915 establishments in Northern Rhodesia 1946 disestablishments in Africa Educational institutions disestablishe…

Ending:

> …Girls' School Mbereshi Girls' School was a mission boarding school at Mbereshi. As "the earliest girls' school in Northern Rhodesia ... this school gained an international reputation." Mbereshi Girls' Boarding School was founded by the missionary Mabel Shaw in 1915, and Shaw served as its Principal until 1940. In 1946 the school was combined with the boy's boarding school to form a new coeducational institution. Alumni Betty Kaunda (1928-2012), First Lady of Zambia References Girls' schools in Zambia Educational institutions established in 1915 Boarding schools in Zambia 1915 establishments in Northern Rhodesia 1946 disestablishments in Africa Educational institutions disestablished in 1946

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Enranger (`rec_eae058dd849b13ab78da04fb`)

Band/weight: B2 / 1.0; caps: none; flags: none

Chunk: 1/1; end boundary: paragraph; characters: 928; paragraphs: 9; alpha: 0.750; repeated trigrams: 0.035

Beginning:

> Enranger The "Enranger" () is the first brand created by Weichai in the light vehicle business field. So far, Weichai's products in the light vehicle field will be under the "Yingzhi" brand, which also marks the official launch of Weichai Motors' Weichai Automobile Co., Ltd. The path of exploration and innovation in the light vehicle market. Production started in the second half of 2014 in an individual facility in Chongqing. Initial capacity was 100,000 cars per year which was planned to later be expanded to 300,000 annually. Models Current Enranger products include the following: Enranger 727 - compact MPV (lower trim level of the 737) Enranger 737 - compact MPV (codenamed M301) Enranger…

Ending:

> …launch of Weichai Motors' Weichai Automobile Co., Ltd. The path of exploration and innovation in the light vehicle market. Production started in the second half of 2014 in an individual facility in Chongqing. Initial capacity was 100,000 cars per year which was planned to later be expanded to 300,000 annually. Models Current Enranger products include the following: Enranger 727 - compact MPV (lower trim level of the 737) Enranger 737 - compact MPV (codenamed M301) Enranger 737 EV - electric compact MPV Enranger EX1 - mini crossover Enranger G5 - compact crossover based on the 737 Enranger P80 - pickup truck Former Enranger G3 - subcompact crossover (code named S201) References Weichai Group

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Women in warfare (1500–1699) (`rec_eba81e0ce96ca6691b222539`)

Band/weight: B2 / 1.0; caps: none; flags: none

Chunk: 2/4; end boundary: paragraph; characters: 5,200; paragraphs: 2; alpha: 0.762; repeated trigrams: 0.044

Beginning:

> Women in warfare (1500–1699) 1550–1599 1546: Second Siege of Diu. Isabel Madeira (captain), Catarina Lopes, Garcia Rodrigues, Isabel Fernandes, and Isabel Dias, formed a group of female combatants who fought in front of the battle against the Turks. 1550s: Siena, Italy falls under siege. Every able citizen was mobilized in the effort to build fortifications, and Laudomia Forteguerri leads a group of 1,000 noble and artisan women to aid in the construction. 1555: Zhuang Chinese woman Wa Shi leads troops into battle on behalf of the Ming Dynasty. 1557: Wa Shi leads over 6000 Zhuang infantry against pirates and successfully defeated them at Wangjiangjing (north of modern Jiaxing). She personal…

Ending:

> …f Suriagehara and Battle of Koriyama. 1587: Catharina Rose commands a women's battalion at the Spanish siege of Sluis in Flanders. 1587: An unnamed woman served in the guise of a man in the Dutch army. 1589: Maria Pita aids in the defence of Corunna against the English armada. 1589: An unnamed woman served in the guise of a man in the Dutch army. 1590: Kaihime led 200 cavalry men in the Siege of Oshi against the Toyotomi clan in the Odawara campaign. 1590: Françoise de Cezelli defeats the Spanish army during the battle of Leucate 1595: Indian Queen Chand Bibi fights the Mughals. 1597: Ebba Stenbock leads the defense of the Turku Castle in Finland after the death of its governor, her spouse.

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

### band_B3

#### 1995 Algerian presidential election (`rec_8b494487569d6b97f95fcf3e`)

Band/weight: B3 / 1.15; caps: none; flags: none

Chunk: 1/1; end boundary: paragraph; characters: 1,200; paragraphs: 7; alpha: 0.798; repeated trigrams: 0.038

Beginning:

> 1995 Algerian presidential election Presidential elections were held in Algeria on 16 November 1995, in the midst of the Algerian Civil War. The result was a victory for Liamine Zeroual, head of the High Council of State at the time, who won 61% of the vote. The Armed Islamic Group of Algeria threatened to kill anyone who voted, with the slogan "one vote, one bullet", but official voter turnout was 74.9%. Candidates Liamine Zeroual, independent Mahfoud Nahnah, candidate of the Islamist Movement of Society for Peace (MSP) Said Sadi, candidate of the secularist Rally for Culture and Democracy Noureddine Boukrouh, candidate of the Party of Algerian Renewal (PRA) Conduct Delegations of observer…

Ending:

> …ment of Society for Peace (MSP) Said Sadi, candidate of the secularist Rally for Culture and Democracy Noureddine Boukrouh, candidate of the Party of Algerian Renewal (PRA) Conduct Delegations of observers came from the Arab League, the African Union, and the United Nations, and reported no major problems. The Armed Islamic Group had threatened to kill voters, but the elections passed with few attacks. Voter turnout was high, despite the three largest parties of the 1991 parliamentary elections (the Islamic Salvation Front, National Liberation Front and Socialist Forces Front) calling for a boycott. Results References Algerian Civil War Presidential elections in Algeria Presidential Algeria

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### The Skeptics' Guide to the Universe (book) (`rec_ed5c0a7706e1c08a0d270f3b`)

Band/weight: B3 / 1.15; caps: none; flags: none

Chunk: 1/1; end boundary: paragraph; characters: 1,658; paragraphs: 10; alpha: 0.798; repeated trigrams: 0.121

Beginning:

> The Skeptics' Guide to the Universe (book) The Skeptics' Guide to the Universe: How to Know What's Really Real in a World Increasingly Full of Fake is a 2018 book meant to be an all-encompassing guide to skeptical thinking written by Steven Novella and co-authored by other hosts of The Skeptics' Guide to the Universe podcastBob Novella, Cara Santa Maria, Jay Novella, and Evan Bernstein. It also contains material from former co-host Perry DeAngelis. About In 2017, Skeptical Inquirer reported that The Skeptics' Guide to the Universe was under development with an expected release in 2018. It became available for pre-order in early 2018, and was released by Grand Central Publishing on October 2…

Ending:

> …view with The European Skeptics Podcast, Jay Novella described their approach to writing the book from the "point of view of an alien species observing the earth from a skeptical perspective using critical thinking," reminiscent of the book's namesake The Hitchhiker's Guide to the Galaxy by Douglas Adams. Reception The book received a favorable review from Kirkus Reviews and was a USA Today bestseller. Publishers Weekly reviewed the book, stating: The book was also reviewed by Rob Palmer for Skeptical Inquirer, who wrote: Author gallery References 2018 non-fiction books English-language books Science books Scientific skepticism Scientific skepticism mass media Grand Central Publishing books

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Loyada, Paschim Medinipur (`rec_6d398a4c67ea06ede18f06da`)

Band/weight: B3 / 1.15; caps: none; flags: none

Chunk: 1/1; end boundary: paragraph; characters: 2,177; paragraphs: 15; alpha: 0.769; repeated trigrams: 0.072

Beginning:

> Loyada, Paschim Medinipur Loyada (also spelled Lowada) is a village in the Debra CD block in the Kharagpur subdivision of the Paschim Medinipur district in the state of West Bengal, India. Geography Location Lowada is located at . Area overview Kharagpur subdivision, shown partly in the map alongside, mostly has alluvial soils, except in two CD blocks in the west – Kharagpur I and Keshiary, which mostly have lateritic soils. Around 74% of the total cultivated area is cropped more than once. With a density of population of 787 per km2nearly half of the district's population resides in this subdivision. 14.33% of the population lives in urban areas and 86.67% lives in the rural areas. Note: T…

Ending:

> …a Balika Vidyalaya is a Bengali-medium girls only institution established in 1971. The school has facilities for teaching from class V to class XII. It has a library with 534 books and 4 computers. Culture David J. McCutchion mentions: The Gopinath temple of the Mukherjee family as standard West Bengal type pancha-ratna, brick temple with terracotta, built in 1805 The Sridhara temple of the Mukherjee family as a flat roofed or chandni type, with terracotta and stucco work The Radha-Govinda temple as a smooth rekha deul with a porch having low-founded pyramidal roof, built in 1860, having rich terracotta. Laoada picture gallery References External links Villages in Paschim Medinipur district

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### The Sky Is Falling (Pearson novel) (`rec_9ffdfc5718a7a4b91cae894d`)

Band/weight: B3 / 1.15; caps: none; flags: none

Chunk: 1/1; end boundary: paragraph; characters: 2,953; paragraphs: 8; alpha: 0.804; repeated trigrams: 0.045

Beginning:

> The Sky Is Falling (Pearson novel) The Sky is Falling is a 1989 young adult novel by Kit Pearson. It is the first novel in the Guests of War trilogy, which follows the lives of Norah and Gavin Stoakes after they are evacuated from England to Canada during World War II. The novel won the Canadian Library Association Book of the Year Award for Children and the Geoffrey Bilson Award (for best Canadian work of historical fiction written for youth). Plot summary Norah and Gavin Stoakes live in a peaceful English village until World War II causes them to be evacuated to Toronto. Norah, an independent ten-year-old, is angry with the evacuation and resents having to care for Gavin. Five-year-old Ga…

Ending:

> …Life begins to improve and Norah accepts Canada as her temporary home for the duration of the war. Title meaning The title of The Sky is Falling comes from a child's misinterpretation of the Blitz. Early in the novel, Norah and Gavin are listening to stories with other evacuees. The group hears the story of The Sky is Falling (Chicken Little), about a chicken who believed the sky was falling. A young evacuee states that is what was happening in England. References 1989 Canadian novels Children's historical novels Canadian children's novels Canadian young adult novels Novels set in Toronto Novels set during World War II Children's books set in Toronto Children's books set during World War II

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### The Adventures of Rocky and Bullwinkle and Friends (`rec_69a1314bf58264a256ff275c`)

Band/weight: B3 / 1.15; caps: none; flags: none

Chunk: 3/7; end boundary: paragraph; characters: 7,837; paragraphs: 5; alpha: 0.799; repeated trigrams: 0.011

Beginning:

> The Adventures of Rocky and Bullwinkle and Friends Episodes were introduced with one of four opening sequences: Rocky flies about snow-covered mountains. Below him, hiking on a snowy trail, Bullwinkle is distracted by a billboard featuring his name, and walks off a ledge. He becomes a large snowball as he rolls downhill. Rocky flies to him and pushes against the snowball, slowing it to a halt at the edge of another cliff. Bullwinkle pops out of the snowball to catch the teetering squirrel at the cliff edge. In a circus, Rocky is preparing to jump from a high diving board into a tub of water tended by Bullwinkle. However, when Rocky jumps, he ends up flying around the circus tent, while Bull…

Ending:

> …mander McBragg", short features on revisionist history as the title character would have imagined it; this was actually prepared for Tennessee Tuxedo and His Tales (and later shown on The Underdog Show). Although the shorts were animated by the same animation company, Gamma Productions, they were produced for Total Television, rather than Ward Productions. These segments were packaged with pre-1990 syndicated versions of The Bullwinkle Show and appear in syndicated episodes of The Underdog Show, Dudley Do Right and Friends, and Uncle Waldo's Cartoon Show. Since 1990, this feature has been divorced from the Bullwinkleverse, and it has never been included in Bullwinkle home videos. Voice cast

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

### band_B4

#### Pennsylvania Department of Education (`rec_c113cc567a6b3d06f85a69a4`)

Band/weight: B4 / 1.25; caps: none; flags: none

Chunk: 1/1; end boundary: paragraph; characters: 4,001; paragraphs: 14; alpha: 0.830; repeated trigrams: 0.078

Beginning:

> Pennsylvania Department of Education The Pennsylvania Department of Education is the executive department of the state charged with publicly funded preschool, K-12 and adult educational budgeting, management and guidelines. As the state education agency, its activities are directed by the governor appointed Pennsylvania's Secretary of Education. The agency is headquartered at 333 Market Street in Harrisburg. The Pennsylvania Department of Education oversees 500 public school districts of Pennsylvania, over 170 public charter schools (2019), Career and Technology Centers/Vocational Technical schools, 29 Intermediate Units, the education of youth in State Juvenile Correctional Institutions, a…

Ending:

> …State Board of Education Professional Standards and Practices Commission Office of Food and Nutrition Programs Special Education Advisory Panel State Boards of Private Schools Power Library Power Library is the online portal to Pennsylvania libraries, a service of the Office of Commonwealth Libraries, Pennsylvania Department of Education. Secretaries of Education See also List of Pennsylvania state agencies State education agency References External links Official website 1837 establishments in Pennsylvania Educational administration Government agencies established in 1837 Education, Department of Department State agencies of Pennsylvania State departments of education of the United States

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### W. Reid Blair (`rec_82800971f33165a4dca3a291`)

Band/weight: B4 / 1.25; caps: none; flags: none

Chunk: 1/1; end boundary: paragraph; characters: 5,580; paragraphs: 15; alpha: 0.783; repeated trigrams: 0.054

Beginning:

> W. Reid Blair William Reid Blair, DVS (June 7, 1875March 3, 1949), better known as W. Reid Blair, worked at the New York Zoological Park (managed by the New York Zoological Society, now the Wildlife Conservation Society) from 1902 to 1940. He began as Assistant Veterinarian and Pathologist and retired from the Zoo as its Director. During his 38-year career at the Zoo, he implemented many advancements in the care of captive animals, and he focused on the educational capacity of zoos. Additionally, it was Dr. Blair who relaxed the insistence of William T. Hornaday, the Zoo's first director, on the use of the formal name "New York Zoological Park" in favor of the more familiar "Bronx Zoo." Ear…

Ending:

> …ls: An Unconventional History of the New York Zoological Society. New York: Harper & Row, 1974. "Bring 'Em Up Alive" [series]. New York World-Telegram, June 1940. Crandall, Lee S. "W. Reid Blair. In Memoriam." Animal Kingdom 52.2 (1949): 59-60. Leister, Claude W. Present Day Mammals. New York: New York Zoological Society, 1931. References External links Blair collection finding aid for collection held by the Wildlife Conservation Society Archives 1875 births 1949 deaths Physicians from Philadelphia McGill University alumni American zoologists American veterinarians Male veterinarians United States Army personnel of World War I United States Army officers Wildlife Conservation Society people

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### ABC News Live (`rec_373ea4b0073f3046b8fc04ba`)

Band/weight: B4 / 1.25; caps: none; flags: none

Chunk: 1/1; end boundary: paragraph; characters: 7,283; paragraphs: 19; alpha: 0.789; repeated trigrams: 0.047

Beginning:

> ABC News Live ABC News Live (a.k.a. ABCNL) is an American streaming video news channel for breaking news, live events, newscasts, and longer-form reports and documentaries operated by ABC News since 2018. The channel is available through various streaming device apps such as Roku, Hulu, YouTube TV, Sling TV, Pluto TV, Xumo, FuboTV, Haystack News, Samsung TV Plus, and the news division's other streaming platforms. Justin Dial is the senior executive director of ABC News Live. History As ABC News Now After having attempted a 20-minute online news program three times a week hosted by Sam Donaldson in 1999, ABC News launched a forerunner of ABC News Now (ABCNN) in March 2003. The service was fe…

Ending:

> …didates are interviewed by three voters and moderated by anchors and correspondents around a table. Episodes would be used as a part of that night's Nightline. In the first episode, Byron Pitts moderates Beto O’Rourke with the second episode being Linsey Davis moderating Senator Cory Booker. Guardians of the Amazon (February 2020) - a documentary regarding rainforest destruction, produced by the Nightline team. References Now Television channels and stations established in 2004 Television channels and stations established in 2018 Television networks in the United States Internet properties established in 2018 Internet television channels 24-hour television news channels in the United States

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Leibniz's notation (`rec_605004799e60d1c47d5313dd`)

Band/weight: B4 / 1.25; caps: none; flags: none

Chunk: 1/2; end boundary: paragraph; characters: 7,751; paragraphs: 28; alpha: 0.807; repeated trigrams: 0.043

Beginning:

> Leibniz's notation In calculus, Leibniz's notation, named in honor of the 17th-century German philosopher and mathematician Gottfried Wilhelm Leibniz, uses the symbols and to represent infinitely small (or infinitesimal) increments of and , respectively, just as and represent finite increments of and , respectively. Consider as a function of a variable , or = . If this is the case, then the derivative of with respect to , which later came to be viewed as the limit was, according to Leibniz, the quotient of an infinitesimal increment of by an infinitesimal increment of , or where the right hand side is Joseph-Louis Lagrange's notation for the derivative of at . The infinitesimal increments a…

Ending:

> …However, an alternative Leibniz notation for higher order derivatives allows for this. This notation was, however, not used by Leibniz. In print he did not use multi-tiered notation nor numerical exponents (before 1695). To write for instance, he would write , as was common in his time. The square of a differential, as it might appear in an arc length formula for instance, was written as . However, Leibniz did use his notation as we would today use operators, namely he would write a second derivative as and a third derivative as . In 1695 Leibniz started to write and for and respectively, but l'Hôpital, in his textbook on calculus written around the same time, used Leibniz's original forms.

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Eirik Kristoffersen (`rec_8537fe22c90a60af14ee0269`)

Band/weight: B4 / 1.25; caps: none; flags: none

Chunk: 1/2; end boundary: paragraph; characters: 8,000; paragraphs: 25; alpha: 0.789; repeated trigrams: 0.059

Beginning:

> Eirik Kristoffersen Eirik Johan Kristoffersen (born 3 April 1969) is a Norwegian Army General who serves as the head of the Norwegian Armed Forces. He is a former Chief of the Norwegian Army and Norwegian Home Guard, and Chief of the Armed Forces' Special Command (FSK). Kristoffersen is the first Norwegian Chief of Defence since World War II, with battle experience. He was awarded the War Cross with Sword in 2011 for his service in Afghanistan. Military career Kristoffersen enrolled in non-commissioned officers' in 1988 and served as squad leader in the Engineer Battalion. After a few months studying engineering in college, he returned to military service in 1989 and served as squad leader…

Ending:

> …988–1989 Non-commissioned officers' training school, 1992–1995 Military Academy, Army, 2008–2009 USMC Command and Staff College, 2014–2015 United States Army War College. Authorship Kristoffersen wrote the book Jegerånden- Å lede i fred, krise og krig [ The Jäger spirit – to lead in peace, crisis and war] (2020). Awards and decorations Kristoffersen is one of Norway's most highly decorated soldiers still on active duty. He has received the following awards: Norwegian medals and awards Foreign decorations Other awards Personal life Kristoffersen lives with his wife Linn-Therece Johansen Kristoffersen and has four children from two former marriages. His interests include hunting and football.

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

### cap_general_short

#### Gürgaletsch (`rec_08f224db7abb45d3c8cc2909`)

Band/weight: B0 / 0.25; caps: ['general_short']; flags: ['short_document']

Chunk: 1/1; end boundary: paragraph; characters: 300; paragraphs: 5; alpha: 0.843; repeated trigrams: 0.000

Beginning:

> Gürgaletsch The Gürgaletsch is a mountain of the Plessur Alps, located between Churwalden and Tschiertschen in the Swiss canton of Graubünden. References External links Gürgaletsch on Hikr Mountains of the Alps Mountains of Switzerland Mountains of Graubünden Two-thousanders of Switzerland Arosa

Ending:

> Gürgaletsch The Gürgaletsch is a mountain of the Plessur Alps, located between Churwalden and Tschiertschen in the Swiss canton of Graubünden. References External links Gürgaletsch on Hikr Mountains of the Alps Mountains of Switzerland Mountains of Graubünden Two-thousanders of Switzerland Arosa

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Shabandar (`rec_09977c596071602871b82052`)

Band/weight: B0 / 0.25; caps: ['general_short']; flags: ['short_document']

Chunk: 1/1; end boundary: paragraph; characters: 322; paragraphs: 4; alpha: 0.773; repeated trigrams: 0.000

Beginning:

> Shabandar Shabandar (, also Romanized as Sha‘bāndar and Shaban Dar; also known as Shab Bīdār and Shab Dar) is a village in Sepiddasht Rural District, Papi District, Khorramabad County, Lorestan Province, Iran. At the 2006 census, its population was 249, in 50 families. References Populated places in Khorramabad County

Ending:

> Shabandar Shabandar (, also Romanized as Sha‘bāndar and Shaban Dar; also known as Shab Bīdār and Shab Dar) is a village in Sepiddasht Rural District, Papi District, Khorramabad County, Lorestan Province, Iran. At the 2006 census, its population was 249, in 50 families. References Populated places in Khorramabad County

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Hamed Ali (`rec_e63714da15a8653094ee999b`)

Band/weight: B0 / 0.25; caps: ['general_short']; flags: ['short_document']

Chunk: 1/1; end boundary: paragraph; characters: 349; paragraphs: 5; alpha: 0.745; repeated trigrams: 0.055

Beginning:

> Hamed Ali Hamed Ali (born 12 January 1956) is a Saudi Arabian sprinter. He competed in the men's 200 metres at the 1976 Summer Olympics. References External links 1956 births Living people Athletes (track and field) at the 1976 Summer Olympics Saudi Arabian male sprinters Olympic athletes for Saudi Arabia Place of birth missing (living people)

Ending:

> Hamed Ali Hamed Ali (born 12 January 1956) is a Saudi Arabian sprinter. He competed in the men's 200 metres at the 1976 Summer Olympics. References External links 1956 births Living people Athletes (track and field) at the 1976 Summer Olympics Saudi Arabian male sprinters Olympic athletes for Saudi Arabia Place of birth missing (living people)

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Everett Whittingham (`rec_b7c96bf0dbedffb8c7d900a8`)

Band/weight: B0 / 0.25; caps: ['general_short']; flags: ['short_document']

Chunk: 1/1; end boundary: paragraph; characters: 373; paragraphs: 6; alpha: 0.783; repeated trigrams: 0.000

Beginning:

> Everett Whittingham Everett Whittingham (born 25 February 1954) is a Jamaican cricketer. He played in one first-class and three List A matches for the Jamaican cricket team from 1980 to 1985. See also List of Jamaican representative cricketers References External links 1954 births Living people Jamaican cricketers Jamaica cricketers Cricketers from Kingston, Jamaica

Ending:

> Everett Whittingham Everett Whittingham (born 25 February 1954) is a Jamaican cricketer. He played in one first-class and three List A matches for the Jamaican cricket team from 1980 to 1985. See also List of Jamaican representative cricketers References External links 1954 births Living people Jamaican cricketers Jamaica cricketers Cricketers from Kingston, Jamaica

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Amselflue (`rec_fbe9c6bbb6e3d3ed98e792e4`)

Band/weight: B0 / 0.25; caps: ['general_short']; flags: ['short_document']

Chunk: 1/1; end boundary: paragraph; characters: 399; paragraphs: 5; alpha: 0.802; repeated trigrams: 0.000

Beginning:

> Amselflue The Amselflue is a mountain of the Plessur Alps, overlooking Arosa in the canton of Graubünden. The main summit has an elevation of 2,781 metres, while the eastern summit, located directly above the Maienfelder Furgga, is 2,768 metre high. References External links Amselflue on Hikr Mountains of the Alps Mountains of Switzerland Mountains of Graubünden Two-thousanders of Switzerland

Ending:

> Amselflue The Amselflue is a mountain of the Plessur Alps, overlooking Arosa in the canton of Graubünden. The main summit has an elevation of 2,781 metres, while the eastern summit, located directly above the Maienfelder Furgga, is 2,768 metre high. References External links Amselflue on Hikr Mountains of the Alps Mountains of Switzerland Mountains of Graubünden Two-thousanders of Switzerland

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

### cap_general_disambiguation

#### Robert Easton (`rec_64ef973f15e38c345250aa3c`)

Band/weight: B0 / 0.25; caps: ['general_short', 'general_disambiguation']; flags: ['disambiguation_page', 'short_document']

Chunk: 1/1; end boundary: paragraph; characters: 302; paragraphs: 4; alpha: 0.682; repeated trigrams: 0.000

Beginning:

> Robert Easton Robert Easton may refer to: Robert Easton (actor) (1930–2011), American actor and dialect coach Robert Easton (bass) (1898–1987), British bass singer Robert Easton (athlete) (born 1960/61), Canadian Paralympic athlete See also Robert Easton Burns (1805–1863), Canadian lawyer and judge

Ending:

> Robert Easton Robert Easton may refer to: Robert Easton (actor) (1930–2011), American actor and dialect coach Robert Easton (bass) (1898–1987), British bass singer Robert Easton (athlete) (born 1960/61), Canadian Paralympic athlete See also Robert Easton Burns (1805–1863), Canadian lawyer and judge

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Emily Martin (`rec_20f1a3654dd9987e3c3d5b89`)

Band/weight: B0 / 0.25; caps: ['general_short', 'general_disambiguation']; flags: ['disambiguation_page', 'short_document']

Chunk: 1/1; end boundary: paragraph; characters: 378; paragraphs: 3; alpha: 0.762; repeated trigrams: 0.000

Beginning:

> Emily Martin Emily Martin may refer to: Emily Martin (1884–1962), aka Emily Dutton, South Australian musician and socialite Emily Martin (anthropologist) (born 1944), sinologist, anthropologist, and feminist Emily Martin (rower) (born 1979), Australian rower Emily Martin (diver), British diver Emily Winfield Martin, American artist and author-illustrator of children's books

Ending:

> Emily Martin Emily Martin may refer to: Emily Martin (1884–1962), aka Emily Dutton, South Australian musician and socialite Emily Martin (anthropologist) (born 1944), sinologist, anthropologist, and feminist Emily Martin (rower) (born 1979), Australian rower Emily Martin (diver), British diver Emily Winfield Martin, American artist and author-illustrator of children's books

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### One Mississippi (`rec_33eb25d6ec7326e5ef50644a`)

Band/weight: B0 / 0.25; caps: ['general_disambiguation']; flags: ['disambiguation_page']

Chunk: 1/1; end boundary: paragraph; characters: 539; paragraphs: 3; alpha: 0.722; repeated trigrams: 0.136

Beginning:

> One Mississippi One Mississippi may refer to: One Mississippi (Brendan Benson album), 1996 One Mississippi (J Church album), 2000 One Mississippi (TV series), a 2016 American television series "One Mississippi", a song on the 2003 album Jillbilly by Jill King "One Mississippi", a song on the 2013 album Bring You Back by Brett Eldredge "One Mississippi", a song on the 2017 album So Good by Zara Larsson "One Mississippi", a song on the 2020 album My Mississippi Reunion by Steve Azar "One Mississippi" (song), a 2021 song by Kane Brown

Ending:

> One Mississippi One Mississippi may refer to: One Mississippi (Brendan Benson album), 1996 One Mississippi (J Church album), 2000 One Mississippi (TV series), a 2016 American television series "One Mississippi", a song on the 2003 album Jillbilly by Jill King "One Mississippi", a song on the 2013 album Bring You Back by Brett Eldredge "One Mississippi", a song on the 2017 album So Good by Zara Larsson "One Mississippi", a song on the 2020 album My Mississippi Reunion by Steve Azar "One Mississippi" (song), a 2021 song by Kane Brown

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Nandu (`rec_a8375516c8d28716960d8a0b`)

Band/weight: B0 / 0.25; caps: ['general_disambiguation']; flags: ['disambiguation_page']

Chunk: 1/1; end boundary: paragraph; characters: 830; paragraphs: 6; alpha: 0.748; repeated trigrams: 0.052

Beginning:

> Nandu Nandu may refer to: Places Chengdu, a city in Sichuan, China, known as (Southern Capital or Nandu) during the early Tang dynasty Jiangling County, a city in Hubei, China, formerly known as (Southern Capital or Nandu) during the later Tang dynasty Nandu River, Hainan province, China Other uses Ñandú, a native South American name for any of three species of Rhea. Nandu (film), a 1981 Tamil film Ñandú (vehicle), a 1940s all-terrain vehicle military vehicle Southern Metropolis Daily, often shortened to Nandu (南都) One of the Argentine Air Force flights that attacked the British fleet in the Battle of San Carlos, during the Falklands War, 1982 People with the given name Nandu Bhende (c. 195…

Ending:

> …dynasty Jiangling County, a city in Hubei, China, formerly known as (Southern Capital or Nandu) during the later Tang dynasty Nandu River, Hainan province, China Other uses Ñandú, a native South American name for any of three species of Rhea. Nandu (film), a 1981 Tamil film Ñandú (vehicle), a 1940s all-terrain vehicle military vehicle Southern Metropolis Daily, often shortened to Nandu (南都) One of the Argentine Air Force flights that attacked the British fleet in the Battle of San Carlos, during the Falklands War, 1982 People with the given name Nandu Bhende (c. 1955–2014), Indian singer Nandu M. Natekar (1933–2021), Indian badminton player See also Nandhu (born 1965), Malayalam film actor

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Old Post Office (`rec_a5be1bf877ab30ee8374ccc4`)

Band/weight: B0 / 0.25; caps: ['general_disambiguation', 'general_linewise_list']; flags: ['disambiguation_page', 'linewise_list']

Chunk: 1/1; end boundary: paragraph; characters: 4,853; paragraphs: 8; alpha: 0.790; repeated trigrams: 0.342

Beginning:

> Old Post Office Old Post Office, or Former Post Office, may refer to: Serbia Old Post Office (Belgrade) United Kingdom Old Post Office, Bristol Tintagel Old Post Office, Tintagel United States (ordered by state and city) Old Athens, Alabama Main Post Office in Athens, Alabama, listed on the National Register of Historic Places Old Brick Post Office in Wickenburg, Arizona, NRHP-listed Old Camden Post Office in Camden, Arkansas, listed on the NRHP in Arkansas Old Post Office (Fayetteville, Arkansas), listed on the NRHP in Arkansas Old Post Office (Hot Springs, Arkansas), listed on the NRHP in Arkansas Little Rock U.S. Post Office and Courthouse, also known as the Old Post Office and Courthous…

Ending:

> …ria County, Texas Old Post Office (Washington, D.C.), NRHP-listed as "Old Post Office and Clock Tower" Old Post Office (Pullman, Washington), NRHP-listed as "U.S. Post Office-Pullman" Old Morgantown Post Office, part of the Monongalia Arts Center in Morgantown, West Virginia, NRHP-listed Old Ashland Post Office, listed on the NRHP in Ashland, Wisconsin Former United States Post Office (Kaukauna, Wisconsin), listed on the NRHP in Wisconsin See also Post office Postal service List of United States post offices Federal Building and Post Office (disambiguation) U.S. Post Office and Courthouse (disambiguation) Postal service (disambiguation) Post Office (disambiguation) Old Post (disambiguation)

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

### cap_general_structured_low_prose

#### Futsal at the 2007 Asian Indoor Games (`rec_3abd80862d172aff4a883a6d`)

Band/weight: B0 / 0.25; caps: ['general_structured_low_prose']; flags: ['duplicate_lines']

Chunk: 1/1; end boundary: paragraph; characters: 650; paragraphs: 31; alpha: 0.754; repeated trigrams: 0.165

Beginning:

> Futsal at the 2007 Asian Indoor Games Futsal at the 2007 Asian Indoor Games was held in Macau, China from 26 October to 3 November 2007. Medalists Medal table Results Men Preliminary Group A Group B Group C Group D Kuwait was disqualified from the tournament on 29 October after Kuwait Football Association was suspended by FIFA. Knockout round Quarterfinals Semifinals Bronze medal match Gold medal match Goalscorers Women Preliminary Group A Group B Placing Knockout round Semifinals Bronze medal match Gold medal match Goalscorers References RSSSF 2007 Asian Indoor Games events Indoor Games 2007 2007 Futsal in Macau

Ending:

> Futsal at the 2007 Asian Indoor Games Futsal at the 2007 Asian Indoor Games was held in Macau, China from 26 October to 3 November 2007. Medalists Medal table Results Men Preliminary Group A Group B Group C Group D Kuwait was disqualified from the tournament on 29 October after Kuwait Football Association was suspended by FIFA. Knockout round Quarterfinals Semifinals Bronze medal match Gold medal match Goalscorers Women Preliminary Group A Group B Placing Knockout round Semifinals Bronze medal match Gold medal match Goalscorers References RSSSF 2007 Asian Indoor Games events Indoor Games 2007 2007 Futsal in Macau

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Russian Professional Basketball League Awards (`rec_23303a5a4dac56966854ccd2`)

Band/weight: B0 / 0.25; caps: ['general_structured_low_prose']; flags: ['duplicate_lines']

Chunk: 1/1; end boundary: paragraph; characters: 1,606; paragraphs: 15; alpha: 0.814; repeated trigrams: 0.230

Beginning:

> Russian Professional Basketball League Awards The Russian Professional Basketball League Awards were the awards that were given out by the former top-tier level professional basketball league in Russia, the Russian Professional Basketball League (RPBL). PBL Awards Russian Professional Basketball League (PBL) 2010–11 season awards PBL Regular Season MVP Maciej Lampe (UNICS Kazan) PBL Playoffs MVP Victor Khryapa (CSKA Moscow) PBL All-Symbolic Team PBL First Symbolic Team Patrick Beverley (Spartak St. Petersburg) Keith Langford (Khimki Moscow Region) Henry Domercant (Spartak St. Petersburg) Sergei Monia (Khimki Moscow Region) Maciej Lampe (UNICS Kazan) PBL Second Symbolic Team Marcus Williams…

Ending:

> …L) 2011–12 season awards PBL Regular Season MVP Davon Jefferson (Triumph Lyubertsy) PBL Playoffs MVP Alexey Shved (CSKA Moscow) PBL All-Symbolic Team PBL First Symbolic Team Patrick Beverley (Spartak St. Petersburg) Zoran Planinić (Khimki Moscow Region) Davon Jefferson (Triumph Lyubertsy) Andrei Kirilenko (CSKA Moscow) Jeremiah Massey (Lokomotiv Kuban) PBL Second Symbolic Team Torey Thomas (Spartak Primorye) Vitaly Fridzon (Khimki Moscow Region) Sergey Karasev (Triumph Lyubertsy) Victor Khryapa (CSKA Moscow) Vladimir Veremeenko (UNICS Kazan) See also Russian Gold Basket Awards References External links Russian Professional Basketball League official website awards European basketball awards

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### 1933 Campeonato Carioca (`rec_09f12a8c6fe474031600bf23`)

Band/weight: B0 / 0.25; caps: ['general_structured_low_prose']; flags: ['duplicate_lines']

Chunk: 1/1; end boundary: paragraph; characters: 2,478; paragraphs: 16; alpha: 0.796; repeated trigrams: 0.115

Beginning:

> 1933 Campeonato Carioca In the 1933 season of the Campeonato Carioca, two championships were disputed, each by a different league. AMEA Championship After the 1932 championship, talks began among the seven main clubs of the AMEA league to discuss whether to adopt professionalism, like APEA in São Paulo had done before, or not. However, after the league's statue was first drafted, only América, Bangu and Fluminense accepted it, although they were joined by Vasco da Gama, which reversed its previous position on that matter. The four teams were consequently expelled from AMEA, which was resolved to remain amateur. Later on, Bonsucesso joined them, and CBD took a stance against professionalism,…

Ending:

> …championship for the 6th time. no teams were relegated. Participating teams System The tournament would be disputed in a double round-robin format, with the team with the most points winning the title. Championship LCF Championship The edition of the Campeonato Carioca organized by LCF (Liga Carioca de Football, or Carioca Football League) kicked off on May 7, 1933, and ended on November 15, 1933. Six teams participated. Bangu won the championship for the 1st time. no teams were relegated. Participating teams System The tournament would be disputed in a double round-robin format, with the team with the most points winning the title. Championship References Campeonato Carioca seasons Carioca

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Virginia State Route 143 (`rec_6768e2762e2c882f15a6b216`)

Band/weight: B0 / 0.25; caps: ['general_structured_low_prose']; flags: ['duplicate_lines']

Chunk: 1/1; end boundary: paragraph; characters: 7,048; paragraphs: 14; alpha: 0.772; repeated trigrams: 0.124

Beginning:

> Virginia State Route 143 State Route 143 (SR 143) is a primary state highway in the U.S. state of Virginia. The state highway runs from Camp Peary near Williamsburg east to U.S. Route 258 (US 258) at Fort Monroe in Hampton. SR 143 is a major local thoroughfare on the Virginia Peninsula portion of the Hampton Roads metropolitan area. The state highway is named Merrimac Trail through the independent city of Williamsburg and adjacent portions of York County and James City County. SR 143 follows Jefferson Avenue through the city of Newport News from the Williamsburg area past Virginia Peninsula Regional Jail to near Downtown Newport News. The state highway, which mostly runs northwest–southeast…

Ending:

> …e with I-64 (Hampton Roads Beltway), which US 60 joins to cross Hampton Roads via the Hampton Roads Bridge-Tunnel to Norfolk. SR 143 continues southeast along two-lane County Street, turns southwest onto Libby Street for one block, then turns south on Mellen Street and intersects SR 169 (Mallory Street) within the Phoebus neighborhood. The state highway crosses the Mill Creek estuary as Ingalls Road and reaches its eastern terminus at its junction with US 258's (Mercury Boulevard) northern terminus at the entrance to Fort Monroe. Major intersections References External links Virginia Highways Project: VA 143 143 State Route 143 State Route 143 State Route 143 State Route 143 State Route 143

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### List of Solanales of South Africa (`rec_fb9edc36b8cebe3a3c4bf73f`)

Band/weight: B0 / 0.25; caps: ['general_structured_low_prose', 'general_linewise_list']; flags: ['linewise_list', 'repetitive_language']

Chunk: 3/4; end boundary: line; characters: 7,996; paragraphs: 7; alpha: 0.832; repeated trigrams: 0.436

Beginning:

> List of Solanales of South Africa Lycium Genus Lycium: Lycium acutifolium E.Mey. ex Dunal, endemic Lycium afrum L. endemic Lycium amoenum Dammer, indigenous Lycium arenicola Miers, indigenous Lycium bosciifolium Schinz, indigenous Lycium cinereum Thunb. indigenous Lycium cordatum Mill. accepted as Carissa bispinosa (L.) Desf. ex Brenan, indigenous Lycium ferocissimum Miers, indigenous Lycium gariepense A.M.Venter, indigenous Lycium grandicalyx Joubert & Venter, indigenous Lycium hantamense A.M.Venter, indigenous Lycium hirsutum Dunal, indigenous Lycium horridum Thunb. indigenous Lycium mascarenense A.M.Venter & A.J.Scott, indigenous Lycium oxycarpum Dunal, endemic Lycium pilifolium C.H.Wrig…

Ending:

> …as Solanum laxum Spreng. not indigenous, naturalised Solanum kibweziense Dammer, accepted as Solanum tettense Klotzsch Solanum koniortodes Dammer, accepted as Solanum tettense Klotzsch Solanum kwebense N.E.Br. ex C.H.Wright, accepted as Solanum tettense Klotzsch, indigenous Solanum kwebense N.E.Br. ex C.H.Wright var. acutius Bitter, accepted as Solanum tettense Klotzsch Solanum kwebense N.E.Br. ex C.H.Wright var. chondropetalum (Dammer) Bitter, accepted as Solanum tettense Klotzsch Solanum kwebense N.E.Br. ex C.H.Wright var. luederitzii (Schinz) Bitter, accepted as Solanum tettense Klotzsch Solanum kwebense N.E.Br. ex C.H.Wright var. majorifrons Bitter, accepted as Solanum tettense Klotzsch

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

### cap_general_linewise_list

#### William J. Haynes II (`rec_bcd265bca5dccfd325809a88`)

Band/weight: B0 / 0.25; caps: ['general_linewise_list', 'general_category_tail', 'general_table_salvage']; flags: ['category_tail', 'linewise_list', 'short_continuation_chunk', 'table_markup_removed']

Chunk: 3/3; end boundary: paragraph; characters: 306; paragraphs: 2; alpha: 0.824; repeated trigrams: 0.023

Beginning:

> William J. Haynes II 1958 births American lawyers Davidson College alumni General Counsels of the United States Army George W. Bush administration personnel Harvard Law School alumni Living people People associated with Jenner & Block People from Waco, Texas Texas Republicans Torture in the United States

Ending:

> William J. Haynes II 1958 births American lawyers Davidson College alumni General Counsels of the United States Army George W. Bush administration personnel Harvard Law School alumni Living people People associated with Jenner & Block People from Waco, Texas Texas Republicans Torture in the United States

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Wouter van Pelt (`rec_ad403943ec3dae9c9ea73684`)

Band/weight: B0 / 0.25; caps: ['general_linewise_list']; flags: ['linewise_list']

Chunk: 1/1; end boundary: paragraph; characters: 1,285; paragraphs: 5; alpha: 0.770; repeated trigrams: 0.134

Beginning:

> Wouter van Pelt Wouter van Pelt (born 23 April 1968 in Alphen aan den Rijn) is a former Dutch field hockey player, who played 236 international matches for the Netherlands, in which he scored 21 goals. The defender made his debut for the Dutch on 27 March 1989 in a match against England. He played in the Dutch League for HDM and BH & BC Breda. Van Pelt was a member of the Dutch national team that won the golden medal at the 1996 Summer Olympics in Atlanta, Georgia. Four years later, at the 2000 Summer Olympics in Sydney, the Dutch once again won the title, with Van Pelt on board. He stopped playing hockey at top level in 2005. External links Dutch Hockey Federation 1968 births Living people…

Ending:

> …ard. He stopped playing hockey at top level in 2005. External links Dutch Hockey Federation 1968 births Living people Dutch male field hockey players Male field hockey defenders Olympic field hockey players for the Netherlands Field hockey players at the 1992 Summer Olympics Field hockey players at the 1996 Summer Olympics 1998 Men's Hockey World Cup players Field hockey players at the 2000 Summer Olympics Olympic gold medalists for the Netherlands Sportspeople from Alphen aan den Rijn Field hockey players from South Holland Olympic medalists in field hockey Medalists at the 2000 Summer Olympics Medalists at the 1996 Summer Olympics Haagsche Delftsche Mixed players 20th-century Dutch people

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Euratom Treaty (`rec_aeffa797e0524e7b1cb1774a`)

Band/weight: B0 / 0.25; caps: ['general_linewise_list']; flags: ['linewise_list']

Chunk: 1/1; end boundary: paragraph; characters: 2,586; paragraphs: 10; alpha: 0.813; repeated trigrams: 0.096

Beginning:

> Euratom Treaty The Euratom Treaty, officially the Treaty establishing the European Atomic Energy Community, established the European Atomic Energy Community. It was signed on 25 March 1957 at the same time as the Treaty establishing the European Economic Community (EEC Treaty). The Euratom Treaty is less well known because of the lower profile of the organisation that it founded. The EEC has evolved into what is now the European Union, but Euratom has remained much the same as it was in 1957 although it is governed by the institutions of the European Union. It was established with its own independent institutions, but the 1967 Merger Treaty merged the institutions of Euratom and the Europea…

Ending:

> …chnology treaties Treaties of Austria Treaties of Bulgaria Treaties of Belgium Treaties of Croatia Treaties of Cyprus Treaties of the Czech Republic Treaties of Denmark Treaties of Estonia Treaties of Finland Treaties of the French Fourth Republic Treaties of West Germany Treaties of Greece Treaties of Hungary Treaties of Ireland Treaties of Italy Treaties of Latvia Treaties of Lithuania Treaties of Luxembourg Treaties of Malta Treaties of the Netherlands Treaties of Poland Treaties of Portugal Treaties of Romania Treaties of Slovakia Treaties of Slovenia Treaties of Spain Treaties of Sweden Treaties of the United Kingdom Treaties extended to Åland March 1957 events in Europe Events in Rome

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### 2010 Ghana Movie Awards (`rec_29b7c4472fe2de5ea727dbf0`)

Band/weight: B0 / 0.25; caps: ['general_linewise_list']; flags: ['linewise_list']

Chunk: 1/1; end boundary: paragraph; characters: 4,884; paragraphs: 30; alpha: 0.759; repeated trigrams: 0.176

Beginning:

> 2010 Ghana Movie Awards The 2010 Ghana Movie Awards was the maiden edition of the ceremony to reward cinematic achievement in Ghana Film Industry. The event was held at the Golden Tulip Hotel, Accra on 25 December 2010. Sinking Sands, Juliet Ibrahim, Nadia Buari, Yvonne Okoro, Majid Michel, John Dumelo & Genevieve Nnaji were among the winners. Awards Best Actor in a Leading Role (English) Senanu Gbedawu (Check Mate) Majid Michel (The Beast) J.O.T Agyemany (I Sing of a Well) Prince Osei (Kiss Me If You Can) Eddie Nartey (Kiss Me If You Can) Van Vicker (Dna Test) Ruffy Samuel (Love & Lust) Best Actress in a Leading Role (English) Martha Ankomah (Kiss Me If You Can) Akorfa Edjeani Asiedu (I Si…

Ending:

> …ng Sands) Ramsey Nouah (Guilty Pleasures) Desmond Elliot (Guilty Pleasures) Uti Nwachukwu (Busting Out (film)) Best Actress - West Africa Collaboration Genevieve Nnaji (Silent Scandals) Nse Ikpe Etim (Guilty Pleasures (2009 film)) Tonto Dikeh (Love & Lust) Uche Jombo (Nollywood Hustlers) Omotola Jalade Ekeinde (Private Storm) Mercy Johnson (Shakira) Best Movie - African Collaboration Sinking Sands Guilty Pleasures (2009 film) Love & Lust Private Storm Bursting Out (film) Best Movie Score The Game (2010 film) Ama Ghana A Sting in a Tale Kiss Me If You Can 4 Play (film) Sinking Sands Favorite Actor Kofi Adjorlolo Favorite Actress Yvonne Nelson References Ghana Movie Awards Ghana 2010 in Ghana

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Khaliji (music) (`rec_b31f126054d03adf1e205d4b`)

Band/weight: B0 / 0.25; caps: ['general_linewise_list']; flags: ['linewise_list']

Chunk: 1/2; end boundary: paragraph; characters: 7,998; paragraphs: 44; alpha: 0.826; repeated trigrams: 0.015

Beginning:

> Khaliji (music) Khaliji music (also spelled Khaleeji; meaning Gulf music) is the music of Eastern Arabia, the Arab states of the Persian Gulf and it is popular across the Arab world. It is traditionally characterized by heavy use of the rebab, oud and other string instruments such as the violin, the occasional use of habbān, and the inclusion of percussion instruments such as the mirwas, tabl, and duff drums. Khaliji music first started as a bedouin tradition with poetry sung by a tribe's shaa'ir, which means poet, usually accompanied by a rebab, the lyrics dealt with tales of honor, love, camel riders, and glory warriors. Khaliji music has roots going back more than 1,000 years, to the Isl…

Ending:

> …i Cheb Faris El Sataifi Spain Hakim (Spanish singer) Greece Grigoris Asikis Konstantinos Argyros Yiorgos Batis George Dalaras Anestis Delias Stratos Pagioumtzis Giorgos Xylouris Babis Tsertos Mariza Koch United Kingdom Yusuf Islam Sami Yusuf Iran Evin Agassi Nematollah Aghasi Hooshmand Aghili Salar Aghili Morteza Ahmadi Alireza Assar Davood Azad Mohsen Chavoshi Farman Fathalian Farzad Fattahi Babak Jahanbakhsh Shahrum Kashani Ehsan Khajeh Amiri Ali Lohrasbi Morteza Pashaei Rahim Shahriari Reza Yazdani Mohsen Yeganeh Sima Bina Leila Forouhar Googoosh Mahasti Marjan (singer) Giti Pashaei Israel Etti Ankri Zohra Al Fassiya Yael Naim Cyprus Hovig Demirjian See also Arabic music Arabic pop music

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

### cap_general_category_tail

#### Caning (`rec_4e5c950ad8c84349677fea8a`)

Band/weight: B0 / 0.25; caps: ['general_category_tail']; flags: ['category_tail', 'short_continuation_chunk']

Chunk: 3/3; end boundary: paragraph; characters: 77; paragraphs: 2; alpha: 0.883; repeated trigrams: 0.000

Beginning:

> Caning BDSM activities Corporal punishments School punishments Whipping Pain

Ending:

> Caning BDSM activities Corporal punishments School punishments Whipping Pain

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Clinton Correctional Facility (`rec_3a346220c348c5a296b5df6b`)

Band/weight: B0 / 0.25; caps: ['general_category_tail']; flags: ['category_tail', 'short_continuation_chunk']

Chunk: 2/2; end boundary: paragraph; characters: 190; paragraphs: 2; alpha: 0.795; repeated trigrams: 0.154

Beginning:

> Clinton Correctional Facility Buildings and structures in Clinton County, New York Capital punishment in New York (state) Prisons in New York (state) 1845 establishments in New York (state)

Ending:

> Clinton Correctional Facility Buildings and structures in Clinton County, New York Capital punishment in New York (state) Prisons in New York (state) 1845 establishments in New York (state)

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Thomas Goode (merchant) (`rec_c788e2ff96642f0883fadb2e`)

Band/weight: B0 / 0.25; caps: ['general_category_tail']; flags: ['category_tail', 'short_continuation_chunk']

Chunk: 3/3; end boundary: paragraph; characters: 239; paragraphs: 3; alpha: 0.808; repeated trigrams: 0.000

Beginning:

> Thomas Goode (merchant) References 1816 births 1882 deaths People from Goolwa, South Australia Settlers of South Australia English emigrants to colonial Australia 19th-century Australian businesspeople 19th-century English businesspeople

Ending:

> Thomas Goode (merchant) References 1816 births 1882 deaths People from Goolwa, South Australia Settlers of South Australia English emigrants to colonial Australia 19th-century Australian businesspeople 19th-century English businesspeople

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Vatican City during World War II (`rec_8e38949e4ff807df6b3ae67e`)

Band/weight: B0 / 0.25; caps: ['general_linewise_list', 'general_category_tail']; flags: ['category_tail', 'linewise_list', 'short_continuation_chunk']

Chunk: 4/4; end boundary: paragraph; characters: 333; paragraphs: 2; alpha: 0.733; repeated trigrams: 0.153

Beginning:

> Vatican City during World War II Neutral states in World War II Pope Pius XII and World War II World War II national military histories Wars involving Vatican City History of the papacy 1939 in Vatican City 1940 in Vatican City 1941 in Vatican City 1942 in Vatican City 1943 in Vatican City 1944 in Vatican City 1945 in Vatican City

Ending:

> Vatican City during World War II Neutral states in World War II Pope Pius XII and World War II World War II national military histories Wars involving Vatican City History of the papacy 1939 in Vatican City 1940 in Vatican City 1941 in Vatican City 1942 in Vatican City 1943 in Vatican City 1944 in Vatican City 1945 in Vatican City

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Giovanni Hidalgo (`rec_9b8cface3e62ae314760b9ba`)

Band/weight: B0 / 0.25; caps: ['general_linewise_list', 'general_category_tail']; flags: ['category_tail', 'linewise_list', 'short_continuation_chunk']

Chunk: 2/2; end boundary: paragraph; characters: 392; paragraphs: 2; alpha: 0.860; repeated trigrams: 0.000

Beginning:

> Giovanni Hidalgo 1963 births Living people American percussionists American drummers Latin jazz percussionists Conga players Barril players Plenera players Puerto Rican educators Music educators musicians from San Juan, Puerto Rico Planet Drum members Djembe players Batá drummers Timbaleros Bongo players American marimbists Timpanists Tubular bells players Tambourine players Güiro players

Ending:

> Giovanni Hidalgo 1963 births Living people American percussionists American drummers Latin jazz percussionists Conga players Barril players Plenera players Puerto Rican educators Music educators musicians from San Juan, Puerto Rico Planet Drum members Djembe players Batá drummers Timbaleros Bongo players American marimbists Timpanists Tubular bells players Tambourine players Güiro players

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

### cap_general_stat_heavy_list

#### Pelopas Kiato F.C. (`rec_8ccd21a84b2f4d06eb00fc28`)

Band/weight: B0 / 0.25; caps: ['general_stat_heavy_list']; flags: ['low_language_content', 'stat_heavy_list']

Chunk: 1/1; end boundary: paragraph; characters: 465; paragraphs: 6; alpha: 0.409; repeated trigrams: 0.000

Beginning:

> Pelopas Kiato F.C. Pelopas Kiato is a Greek football club, based in Kiato, Corinthia. Honours Domestic Amateur Cup: 1 1990-91 Corinthia FCA Championship: 13 1947-48, 1948–49, 1952–53, 1965–66, 1972–73, 1973–74, 1987–88, 1998–99, 2002–03, 2005–06, 2008–09, 2011–12, 2016–17 Corinthia FCA Cup: 10 1972-73, 1978–79, 1990–91, 1993–94, 1994–95, 1995–96, 1996–97, 2002–03, 2003–04, 2017-18 Association football clubs established in 1926 1926 establishments in Greece

Ending:

> Pelopas Kiato F.C. Pelopas Kiato is a Greek football club, based in Kiato, Corinthia. Honours Domestic Amateur Cup: 1 1990-91 Corinthia FCA Championship: 13 1947-48, 1948–49, 1952–53, 1965–66, 1972–73, 1973–74, 1987–88, 1998–99, 2002–03, 2005–06, 2008–09, 2011–12, 2016–17 Corinthia FCA Cup: 10 1972-73, 1978–79, 1990–91, 1993–94, 1994–95, 1995–96, 1996–97, 2002–03, 2003–04, 2017-18 Association football clubs established in 1926 1926 establishments in Greece

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

### cap_general_table_salvage

#### William J. Haynes II (`rec_bcd265bca5dccfd325809a88`)

Band/weight: B0 / 0.25; caps: ['general_linewise_list', 'general_category_tail', 'general_table_salvage']; flags: ['category_tail', 'linewise_list', 'short_continuation_chunk', 'table_markup_removed']

Chunk: 3/3; end boundary: paragraph; characters: 306; paragraphs: 2; alpha: 0.824; repeated trigrams: 0.023

Beginning:

> William J. Haynes II 1958 births American lawyers Davidson College alumni General Counsels of the United States Army George W. Bush administration personnel Harvard Law School alumni Living people People associated with Jenner & Block People from Waco, Texas Texas Republicans Torture in the United States

Ending:

> William J. Haynes II 1958 births American lawyers Davidson College alumni General Counsels of the United States Army George W. Bush administration personnel Harvard Law School alumni Living people People associated with Jenner & Block People from Waco, Texas Texas Republicans Torture in the United States

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Ordane Kanda-Kanyinda (`rec_3913c643100462f166f04a46`)

Band/weight: B0 / 0.25; caps: ['general_table_salvage']; flags: ['table_markup_removed']

Chunk: 1/1; end boundary: paragraph; characters: 1,671; paragraphs: 15; alpha: 0.771; repeated trigrams: 0.022

Beginning:

> Ordane Kanda-Kanyinda Ordane Kanda-Kanyinda (born 9 September 1996) is a Belgian professional basketball player who last played for Feyenoord Basketball. Professional career Kanda-Kanyinda was born in Antwerp and played in the youth development department of Antwerp Giants, where he started his professional basketball career. He played his first game for Antwerp's first team in the 2013–14 season, in the EuroChallenge campaign. The season 2016–17 was his first year where he was a definitive member of the first Antwerp Giants team. Antwerp finished second in the regular season and lost in the semi-finals of the playoffs. On 3 July 2017, Kanda-Kanyinda was sent on a one-year loan to Forward L…

Ending:

> …quarter-finals, which Rotterdam won 89–92. Rotterdam reached the DBL semi-finals for the first time in 12 years. In October 2018, Kanda signed with Kangoeroes Basket Mechelen. Kanda signed with Spirou for the 2019–20 season. On 4 February 2020, Kanda signed a try-out contract with Heroes Den Bosch. On 17 July 2020, Kanda returned to Rotterdam, now named Feyenoord. Career statistics Domestic leagues |-|- Source: RealGM References 1996 births Living people Antwerp Giants players Belgian men's basketball players Dutch Basketball League players Feyenoord Basketball players Heroes Den Bosch players Kangoeroes Basket Mechelen players Point guards Spirou Charleroi players Sportspeople from Antwerp

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### FC Luch Vladivostok (`rec_c10b78acccaa5c675a3f8945`)

Band/weight: B0 / 0.25; caps: ['general_table_salvage']; flags: ['table_markup_removed']

Chunk: 1/1; end boundary: paragraph; characters: 4,178; paragraphs: 22; alpha: 0.771; repeated trigrams: 0.034

Beginning:

> FC Luch Vladivostok FC Luch Vladivostok () was an association football club based in Vladivostok, Russia. In 2005, Luch won the Russian First Division and played in the Premier League from 2006 to 2008. The club was called Luch-Energiya from 2003 to 2018, when it was renamed due to sponsorship from Dalenergo, an energy distribution company. History Luch has been playing in the Soviet Union championship since 1958. The name Luch means Ray. The club played in the Far East regional tournament of "B-class" teams and eventually won it in 1965, earning promotion to "A-class". Luch played in this regional tournament until league reorganization in 1972. From 1972 to 1991, Luch played in the Eastern…

Ending:

> …nce once a year whereas we have to do it for all away matches". Srđan Radonjić said "It is just crazy, they should have two Russian premier leagues, one for the European teams and another for Asian teams. Vladivostok is 4,000 miles from Moscow." Notable players Had international caps for their respective countries. Players whose name is listed in bold represented their countries while playing for Luch-Energiya. USSR/Russia Former USSR countries Europe Africa References External links Official website Association football clubs established in 1958 Luch Vladivostok Luch Vladivostok 1958 establishments in Russia Association football clubs disestablished in 2020 2020 disestablishments in Russia

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Survivor (Destiny's Child album) (`rec_c05a3d00e49d6ed368735f28`)

Band/weight: B0 / 0.25; caps: ['general_table_salvage']; flags: ['table_markup_removed']

Chunk: 1/6; end boundary: paragraph; characters: 7,414; paragraphs: 9; alpha: 0.791; repeated trigrams: 0.098

Beginning:

> Survivor (Destiny's Child album) Survivor is the third studio album by American girl group Destiny's Child. It was released on April 25, 2001, by Columbia Records. As their breakthrough second studio album The Writing's on the Wall (1999) became a rising commercial success, Destiny's Child faced the controversial departure of original members LeToya Luckett and LaTavia Roberson, who were replaced with Farrah Franklin and Michelle Williams, in February 2000. Soon afterwards, they commenced production of their third studio album, tentatively titled Independent Women. Mere five months after joining, Franklin departed from the group in July, and "Independent Women Part I" was subsequently relea…

Ending:

> …The label kept saying "Do another song, do another song, do another song". It wasn't planned. It wasn't like I said, OK, I'm going to take charge." However, Kelly Rowland and Michelle Williams co-wrote only one track–"Outro (DC-3) Thank You". On July 20, Farrah Franklin departed from Destiny's Child, having already recorded backing vocals for several tracks, including "Independent Women Part I". The group embarked on Christina Aguilera's tour Christina Aguilera in Concert as an opening act on July 31, touring until October and simultaneously recording the album. Six songs, including both parts to "Independent Women", "Nasty Girl" and the unreleased "I Tried", had been recorded by September.

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Nine Inch Nails (`rec_e59aaa26557c7d28a5135eda`)

Band/weight: B0 / 0.25; caps: ['general_table_salvage']; flags: ['table_markup_removed']

Chunk: 1/9; end boundary: paragraph; characters: 7,989; paragraphs: 17; alpha: 0.779; repeated trigrams: 0.047

Beginning:

> Nine Inch Nails Nine Inch Nails, commonly abbreviated as NIN and stylized as NIИ, is an American industrial rock band formed in Cleveland in 1988. Singer, songwriter, multi-instrumentalist, and producer Trent Reznor was the only permanent member of the band until his frequent collaborator, Atticus Ross, joined in 2016. The band's debut album, Pretty Hate Machine (1989), was released via TVT Records. After disagreeing with TVT about how to promote the album, the band signed with Interscope Records and released the EP Broken (1992). The following albums, The Downward Spiral (1994) and The Fragile (1999), were released to critical acclaim and commercial success. Following a hiatus, Nine Inch N…

Ending:

> …en embarked on a world tour that continued through the first Lollapalooza festival in 1991. Broken (1992–1993) After a poor European reception opening for Guns N' Roses, the band returned to the US amid pressure from TVT to produce a follow-up to Pretty Hate Machine. After finding out they were hindering control of his project, Reznor criticized the labeling of Nine Inch Nails as a commercially oriented band and demanded his label terminate his contract, but they ignored his plea. In response, Reznor secretly began recording under various pseudonyms to avoid record company interference. Involved in a feud with TVT, he signed a record deal with Interscope Records and created Nothing Records:

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

### cap_general_sensitive_context_review

#### Alachua County, Florida (`rec_5f3f92850e4e1922a6ae1272`)

Band/weight: B0 / 0.25; caps: ['general_linewise_list', 'general_sensitive_context_review']; flags: ['human_sensitive_context_reviewed', 'linewise_list']

Chunk: 3/3; end boundary: paragraph; characters: 2,466; paragraphs: 13; alpha: 0.817; repeated trigrams: 0.049

Beginning:

> Alachua County, Florida On August 9, 2021, a prison inmate, Erica Thompson, gave birth while being held in the county jail. Her baby died. Despite the mother's screams, jail staff did not provide or call for medical assistance. An investigation held that law enforcement did not violate any law or policy. Landfills Alachua County is the site of five closed landfills—Southwest Landfill, Southeast Landfill, Northwest Landfill, Northeast Landfill, and Northeast Auxiliary Landfill. Since 1999, all solid waste from Alachua County has been hauled to the New River Solid Waste Facility in Raiford, in neighboring Union County. Communities Unincorporated communities Arredondo Bland Campville Cross Cre…

Ending:

> …ville. Spring Grove was the second county seat of Alachua County, after Newnansville was included in the newly created Columbia County, until Newnansville was returned to Alachua County and restored as the county seat. It was abandoned sometime in the middle of the 19th century. See also Alachua County Library District Florida State Parks in Alachua County National Register of Historic Places listings in Alachua County, Florida List of counties in Florida Notes External links Alachua County 1824 establishments in Florida Territory Florida placenames of Native American origin Charter counties in Florida Gainesville metropolitan area, Florida North Florida Populated places established in 1824

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

### boundary_non_paragraph

#### Anagyrus (`rec_203faf92b1b110fe79d99649`)

Band/weight: B0 / 0.25; caps: ['general_linewise_list']; flags: ['linewise_list']

Chunk: 1/2; end boundary: line; characters: 7,985; paragraphs: 3; alpha: 0.705; repeated trigrams: 0.212

Beginning:

> Anagyrus Anagyrus is a large genus of parasitic wasps from the family Encyrtidae. Anagyrus is distributed throughout the world. A subgenus of Anagyrus is known as Nesoanagyrus (Beardsley 1969) Species There are at least 247 species in this genus which consists of: Anagyrus abatos (Noyes & Menezes, 2000) Anagyrus abdulrassouli (Myartseva, Sugonjaev & Trjapitzin, 1982) Anagyrus abyssinicus Compere, 1939 Anagyrus aceris Noyes & Hayat, 1994 Anagyrus aciculatus (Blanchard, 1940) Anagyrus adamsoni Timberlake, 1941 Anagyrus aega Noyes, 2000 Anagyrus aegyptiacus Moursi, 1948 Anagyrus agraensis Saraswat 1975 Anagyrus alami Hayat 1970 Anagyrus albatus Myartseva, 1982 Anagyrus aligarhensis Agarwal & A…

Ending:

> …Domenichini, 1953 Anagyrus semifulvus Girault, 1915 Anagyrus shahidi Hayat, 1979 Anagyrus siccus (Prinsloo & Annecke, 1976) Anagyrus similis (Girault 1915) Anagyrus sinensis Noyes & Hayat, 1994 Anagyrus sinope Noyes & Menezes 2000 Anagyrus smithi Doutt, 1952 Anagyrus sogdianus Sugonjaev, 1968 Anagyrus sophax Noyes & Menezes 2000 Anagyrus spaici (Hoffer, 1970) Anagyrus spica (Girault 1921) Anagyrus subalbipes Ishii, 1928 Anagyrus subflaviceps (Girault 1915) Anagyrus subnigricornis Ishii, 1928 Anagyrus subproximus (Silvestri, 1915) Anagyrus subtilis Noyes & Hayat, 1994 Anagyrus sucro Noyes, 2000 Anagyrus suia Noyes, 2000 Anagyrus surekhae Noyes & Menezes 2000 Anagyrus swezeyi Timberlake, 1919

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### List of Saxifragales of South Africa (`rec_2ec3fb84b0c5e38c5277d166`)

Band/weight: B0 / 0.25; caps: ['general_linewise_list']; flags: ['linewise_list']

Chunk: 3/4; end boundary: line; characters: 7,969; paragraphs: 2; alpha: 0.812; repeated trigrams: 0.342

Beginning:

> List of Saxifragales of South Africa Crassula multiflora Schonland & Baker f. subsp. multiflora, endemic Crassula muricata Thunb. endemic Crassula muscosa L. indigenous Crassula muscosa L. var. muscosa, indigenous Crassula muscosa L. var. obtusifolia (Harv.) G.D.Rowley, indigenous Crassula muscosa L. var. parvula (Eckl. & Zeyh.) Toelken, endemic Crassula muscosa L. var. polpodacea (Eckl. & Zeyh.) G.D.Rowley, endemic Crassula namaquensis Schonland & Baker f. indigenous Crassula namaquensis Schonland & Baker f. subsp. comptonii (Hutch. & Pillans) Toelken, endemic Crassula namaquensis Schonland & Baker f. subsp. lutea (Schonland) Toelken, endemic Crassula namaquensis Schonland & Baker f. subsp…

Ending:

> …enous Crassula thunbergiana Schult. subsp. minutiflora (Schonland & Baker f.) Toelken, indigenous Crassula thunbergiana Schult. subsp. thunbergiana, endemic Crassula tomentosa Thunb. indigenous Crassula tomentosa Thunb. var. glabrifolia (Harv.) G.D.Rowley, indigenous Crassula tomentosa Thunb. var. tomentosa, indigenous Crassula tuberella Toelken, indigenous Crassula umbella Jacq. endemic Crassula umbellata Thunb. endemic Crassula umbraticola N.E.Br. indigenous Crassula vaginata Eckl. & Zeyh. indigenous Crassula vaginata Eckl. & Zeyh. subsp. vaginata, indigenous Crassula vaillantii (Willd.) Roth, not indigenous, naturalised Crassula vestita Thunb. endemic Crassula werneri N.Jacobsen, endemic

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### List of moths of Australia (Cosmopterigidae) (`rec_55072f6f14692e650bb5412f`)

Band/weight: B0 / 0.25; caps: ['general_linewise_list']; flags: ['linewise_list']

Chunk: 1/2; end boundary: line; characters: 7,968; paragraphs: 3; alpha: 0.717; repeated trigrams: 0.153

Beginning:

> List of moths of Australia (Cosmopterigidae) This is a list of the Australian species of the family Cosmopterigidae. It also acts as an index to the species articles and forms part of the full List of moths of Australia. Chrysopeleiinae Cholotis exodroma (Meyrick, 1897) Cholotis semnostola (Meyrick, 1897) Eumenodora encrypta Meyrick, 1906 Ithome lassula Hodges, 1962 Leptozestis anagrapta (Meyrick, 1897) Leptozestis antithetis (Meyrick, 1897) Leptozestis argoscia (Lower, 1904) Leptozestis autochroa (Meyrick, 1915) Leptozestis capnopora (Meyrick, 1897) Leptozestis cataspoda (Meyrick, 1897) Leptozestis charmosyna (Meyrick, 1921) Leptozestis crassipalpis (Turner, 1923) Leptozestis crebra (Meyri…

Ending:

> …Limnaecia stenosticha Turner, 1926 Limnaecia symplecta Turner, 1923 Limnaecia syntaracta Meyrick, 1897 Limnaecia tetraplanetis Meyrick, 1897 Limnaecia triplaneta Meyrick, 1921 Limnaecia trisema Meyrick, 1897 Limnaecia trissodesma (Meyrick, 1887) Limnaecia trixantha (Lower, 1920) Limnaecia xanthopelta Lower, 1903 Limnaecia xanthopis Meyrick, 1920 Limnaecia zonomacula Lower, 1908 Limnaecia zotica Meyrick, 1921 Macrobathra allocrana Turner, 1916 Macrobathra allophyla (Turner, 1944) Macrobathra alternatella (Walker, 1864) Macrobathra anacampta Meyrick, 1914 Macrobathra anemarcha Meyrick, 1886 Macrobathra anemodes Meyrick, 1886 Macrobathra aneurae Turner, 1932 Macrobathra aphristis Meyrick, 1889

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### List of Saxifragales of South Africa (`rec_687ad59a83c4238fecdaad6a`)

Band/weight: B0 / 0.25; caps: ['general_linewise_list']; flags: ['linewise_list']

Chunk: 1/4; end boundary: line; characters: 7,975; paragraphs: 15; alpha: 0.808; repeated trigrams: 0.168

Beginning:

> List of Saxifragales of South Africa Saxifragales (saxifrages) is an order of flowering plants (Angiosperms). They are an extremely diverse group of plants which include trees, shrubs, perennial herbs, succulent and aquatic plants. The degree of diversity in terms of vegetative and floral features makes it difficult to define common features that unify the order. In the Angiosperm Phylogeny Group classification system, the Saxifragales are placed within the major division of flowering plants referred to as eudicots, specifically the core eudicots. This subgroup consists of the Dilleniaceae, superasterids and superrosids. The superrosids in turn have two components, rosids and Saxifragales.…

Ending:

> …indigenous Cotyledon pendens Van Jaarsv. endemic Cotyledon petiolaris Van Jaarsv. endemic Cotyledon rhombifolia Haw. accepted as Adromischus rhombifolius (Haw.) Lem. Cotyledon tomentosa Harv. indigenous Cotyledon tomentosa Harv. subsp. ladismithiensis (Poelln.) Toelken, endemic Cotyledon tomentosa Harv. subsp. tomentosa, endemic Cotyledon velutina Hook.f. endemic Cotyledon woodii Schonland & Baker f. endemic Cotyledon xanthantha Van Jaarsv. & Eggli, endemic Crassula Genus Crassula: Crassula acinaciformis Schinz, indigenous Crassula alba Forssk. var. alba, indigenous Crassula alba Forssk. var. pallida Toelken, indigenous Crassula alba Forssk. var. parvisepala (Schonland) Toelken, indigenous

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### First-wave feminism (`rec_699d0ee0ff69d9b99de4f577`)

Band/weight: B0 / 0.25; caps: ['general_linewise_list']; flags: ['linewise_list']

Chunk: 8/11; end boundary: line; characters: 7,914; paragraphs: 2; alpha: 0.778; repeated trigrams: 0.167

Beginning:

> First-wave feminism 1839 US, Mississippi: Mississippi was the first U.S. state that gave married women limited property rights. United Kingdom: The Custody of Infants Act 1839 made it possible for divorced mothers to be granted custody of their children under seven, but only if the Lord Chancellor agreed to it, and only if the mother was of good character. US, Mississippi: The Married Women's Property Act 1839 granted married women the right to own (but not control) property in their own name. 1840 US, Texas: Married women were allowed to own property in their own name. 1841 Bulgaria: The first secular girls school in Bulgaria was opened, making education and the profession of teacher avail…

Ending:

> …right to vote. New Zealand: New Zealand became the first self-governing country in the world in which all women had the right to vote in parliamentary elections. Cook Islands: The Cook Islands granted women the right to vote in island councils and a federal parliament. 1894 South Australia: South Australia granted women the right to vote. United Kingdom: The United Kingdom extended the right to vote in local elections to married women. 1895 US: Almost all U.S. states had passed some form of Sole Trader Laws, Property Laws, and Earnings Laws, granting married women the right to trade without their husbands' consent, own and/or control their own property, and control their own earnings. 1896

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### List of Saxifragales of South Africa (`rec_ca42c8e9e1bd7d8a72de1a27`)

Band/weight: B0 / 0.25; caps: ['general_linewise_list']; flags: ['linewise_list']

Chunk: 2/4; end boundary: line; characters: 7,963; paragraphs: 2; alpha: 0.815; repeated trigrams: 0.349

Beginning:

> List of Saxifragales of South Africa Crassula alcicornis Schonland, endemic Crassula alpestris Thunb. indigenous Crassula alpestris Thunb. subsp. alpestris, endemic Crassula alpestris Thunb. subsp. massonii (Britten & Baker f.) Toelken, endemic Crassula alstonii Marloth, endemic Crassula ammophila Toelken, endemic Crassula aphylla Schonland & Baker f. endemic Crassula arborea Medik. accepted as Crassula arborescens (Mill.) Willd. subsp. arborescens, indigenous Crassula arborescens (Mill.) Willd. endemic Crassula arborescens (Mill.) Willd. subsp. arborescens, endemic Crassula arborescens (Mill.) Willd. subsp. undulatifolia Toelken, endemic Crassula argentea Thunb. accepted as Crassula ovata…

Ending:

> …s (Haw.) D.Dietr. subsp. hispida (Haw.) Toelken, endemic Crassula mesembryanthemoides (Haw.) D.Dietr. subsp. mesembryanthemoides, endemic Crassula minuta Toelken, endemic Crassula mollis Thunb. endemic Crassula montana Thunb. indigenous Crassula montana Thunb. subsp. montana, endemic Crassula montana Thunb. subsp. quadrangularis (Schonland) Toelken, endemic Crassula multicava Lem. indigenous Crassula multicava Lem. subsp. floribunda Friedrich ex Toelken, endemic Crassula multicava Lem. subsp. multicava, endemic Crassula multiceps Harv. endemic Crassula multiflora Schonland & Baker f. indigenous Crassula multiflora Schonland & Baker f. subsp. leucantha (Schonland & Baker f.) Toelken, endemic

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### First-wave feminism (`rec_ddb6df1b0b05db91f7997f78`)

Band/weight: B0 / 0.25; caps: ['general_linewise_list']; flags: ['linewise_list']

Chunk: 9/11; end boundary: line; characters: 7,948; paragraphs: 2; alpha: 0.800; repeated trigrams: 0.178

Beginning:

> First-wave feminism Argentina: A group of anarcha-feminist women, headed by Virginia Bolten, publish La Voz de la Mujer, one of the first feminist newspapers of Latin America. US, Idaho: Idaho granted women the right to vote. 1900 Western Australia: Western Australia granted women the right to vote. Belgium: Legal majority was granted to unmarried women. Egypt: A school for female teachers was founded in Cairo. France: Women were allowed to practice law. Korea: The post office profession was opened to women. Tunisia: The first public elementary school for girls was opened. Japan: The first women's university was opened. Baden, Germany: Universities opened to women. Sweden: Maternity leave w…

Ending:

> …ondon Federation of Suffragettes. 1913 Russia: In 1913 Russian women observed their first International Women's Day on the last Sunday in February. Following discussions, International Women's Day was transferred to 8 March and this day has remained the global date for International Women's Day ever since. US, Alaska: Alaska granted women the right to vote. Norway: Norway granted women the right to vote. Japan: Public universities opened to women. United Kingdom: The suffragette Emily Davison was killed by the King's horse at The Derby. United Kingdom: 50,000 women taking part in a pilgrimage organized by the National Union of Women's Suffrage Societies arrived in Hyde Park on July 26. 1914

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### List of Solanales of South Africa (`rec_fb9edc36b8cebe3a3c4bf73f`)

Band/weight: B0 / 0.25; caps: ['general_structured_low_prose', 'general_linewise_list']; flags: ['linewise_list', 'repetitive_language']

Chunk: 3/4; end boundary: line; characters: 7,996; paragraphs: 7; alpha: 0.832; repeated trigrams: 0.436

Beginning:

> List of Solanales of South Africa Lycium Genus Lycium: Lycium acutifolium E.Mey. ex Dunal, endemic Lycium afrum L. endemic Lycium amoenum Dammer, indigenous Lycium arenicola Miers, indigenous Lycium bosciifolium Schinz, indigenous Lycium cinereum Thunb. indigenous Lycium cordatum Mill. accepted as Carissa bispinosa (L.) Desf. ex Brenan, indigenous Lycium ferocissimum Miers, indigenous Lycium gariepense A.M.Venter, indigenous Lycium grandicalyx Joubert & Venter, indigenous Lycium hantamense A.M.Venter, indigenous Lycium hirsutum Dunal, indigenous Lycium horridum Thunb. indigenous Lycium mascarenense A.M.Venter & A.J.Scott, indigenous Lycium oxycarpum Dunal, endemic Lycium pilifolium C.H.Wrig…

Ending:

> …as Solanum laxum Spreng. not indigenous, naturalised Solanum kibweziense Dammer, accepted as Solanum tettense Klotzsch Solanum koniortodes Dammer, accepted as Solanum tettense Klotzsch Solanum kwebense N.E.Br. ex C.H.Wright, accepted as Solanum tettense Klotzsch, indigenous Solanum kwebense N.E.Br. ex C.H.Wright var. acutius Bitter, accepted as Solanum tettense Klotzsch Solanum kwebense N.E.Br. ex C.H.Wright var. chondropetalum (Dammer) Bitter, accepted as Solanum tettense Klotzsch Solanum kwebense N.E.Br. ex C.H.Wright var. luederitzii (Schinz) Bitter, accepted as Solanum tettense Klotzsch Solanum kwebense N.E.Br. ex C.H.Wright var. majorifrons Bitter, accepted as Solanum tettense Klotzsch

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### William Weintraub (`rec_ff9dffa9e298302d4622efee`)

Band/weight: B0 / 0.25; caps: ['general_linewise_list']; flags: ['linewise_list']

Chunk: 1/2; end boundary: line; characters: 7,969; paragraphs: 15; alpha: 0.753; repeated trigrams: 0.125

Beginning:

> William Weintraub William Weintraub (February 19, 1926 – November 6, 2017) was a Canadian documentarian/filmmaker, journalist and author, best known for his long career with the National Film Board of Canada (NFB). Early life Weintraub was born in Montreal, to Louis Weintraub and Mina Blumer Weintraub, and grew up in the blue-collar neighbourhood of Verdun. His father had been a stock broker; he lost everything in the Wall Street Crash of 1929 and worked as the manager of a corner store. William studied English Literature and political science at McGill University, where he had worked on the McGill Daily. In 1947, he took the job of a ski reporter at The Montreal Gazette, from which he was…

Ending:

> …rector Background to Latin America - documentary short, James Beveridge 1963 - writer Canada Between Two World Wars - documentary short 1963 - writer and producer Canada: Human Vaccine - documentary short, Hector Lemieux 1963 - writer Canada: Beef Cattle - documentary short, Hector Lemieux 1963 - writer Canada: Calf Leather - documentary short, Hector Lemieux 1963 - writer Canada: Heating Units - documentary short, Hector Lemieux 1963 - writer The Visit - documentary short John Kemeny 1964 - writer Turn of the Century - documentary short 1964 - writer and producer Haida Carver - documentary short, Richard Gilbert 1964 - writer Landfall Asia - documentary short, Gordon Sparling 1964 - writer

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:
