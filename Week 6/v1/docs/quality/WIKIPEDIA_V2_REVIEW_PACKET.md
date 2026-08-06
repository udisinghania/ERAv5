# Wikipedia corpus-v2 review packet

This is the human validation gate between building the Wikipedia-specific quality policy and adapting quality logic to the other six data lanes.

## What to learn from this stage

The signals are measurements, the band is a policy decision derived from those measurements, the weight controls ordinary sampling, and a cap is a final safety ceiling. A cap may be configured without activating.

## Population and weighted supply

- Physical records: 5,144
- Weighted records before caps: 5,195.75
- Review examples: 56

| Band | Physical records |
|---|---:|
| B0 | 738 |
| B1 | 1 |
| B2 | 1,334 |
| B3 | 1,620 |
| B4 | 1,451 |

## Do the caps currently activate?

| Group | Records | Share after weights | Cap | Activates? |
|---|---:|---:|---:|---|
| general_short | 304 | 1.46% | 1.00% | yes |
| general_disambiguation | 106 | 0.51% | 2.00% | no |
| general_structured_low_prose | 26 | 0.13% | 2.00% | no |
| general_linewise_list | 336 | 1.62% | 3.00% | no |
| general_category_tail | 27 | 0.13% | 0.50% | no |
| general_sensitive_context_review | 1 | 0.00% | 0.10% | no |
| all_B0_combined | 738 | 3.55% | 5.00% | no |

Only an activating cap changes the distribution beyond the sampling weights. Non-activating caps remain useful as guards if the corpus grows later.

## Review rubric

For each example, inspect whether it contains meaningful language, is coherent and complete, belongs in its assigned band/cap group, is PII-safe, and ends cleanly. Then choose keep, downweight, or reject. Do not change a threshold after one unusual example; look for a repeated error pattern.

## Deterministic sample

Five examples are drawn across the length range of every band and cap group. All 9 non-paragraph boundary chunks are included. Some records intentionally appear in more than one stratum because they test different claims.

### band_B0

#### Caning (`rec_6f908dd96f503ab796b29af1`)

Band/weight: B0 / 0.25; caps: ['general_category_tail']; flags: ['category_tail', 'short_continuation_chunk']

Chunk: 3/3; end boundary: paragraph; characters: 77; paragraphs: 2; alpha: 0.883; repeated trigrams: 0.000

Beginning:

> Caning BDSM activities Corporal punishments School punishments Whipping Pain

Ending:

> Caning BDSM activities Corporal punishments School punishments Whipping Pain

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Leucania incognita (`rec_466710bb64afaad2cb6ea984`)

Band/weight: B0 / 0.25; caps: ['general_short']; flags: ['short_document']

Chunk: 1/1; end boundary: paragraph; characters: 353; paragraphs: 6; alpha: 0.788; repeated trigrams: 0.019

Beginning:

> Leucania incognita Leucania incognita is a species of cutworm or dart moth in the family Noctuidae first described by William Barnes and James Halliday McDunnough in 1918. It is found in North America. The MONA or Hodges number for Leucania incognita is 10450. References Further reading Leucania Articles created by Qbugbot Moths described in 1918

Ending:

> Leucania incognita Leucania incognita is a species of cutworm or dart moth in the family Noctuidae first described by William Barnes and James Halliday McDunnough in 1918. It is found in North America. The MONA or Hodges number for Leucania incognita is 10450. References Further reading Leucania Articles created by Qbugbot Moths described in 1918

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Wyn Jones (`rec_3752d54999f68604721e77d8`)

Band/weight: B0 / 0.25; caps: ['general_disambiguation']; flags: ['disambiguation_page']

Chunk: 1/1; end boundary: paragraph; characters: 548; paragraphs: 3; alpha: 0.715; repeated trigrams: 0.062

Beginning:

> Wyn Jones Wyn Jones may refer to: Wyn Jones (colonial administrator) (1926–1993), British colonial administrator Wyn Jones (police officer) (born c. 1943), British police officer, Assistant Commissioner of the Metropolitan Police Wyn Jones (rugby union) (born 1992), Welsh rugby union player See also Alun Wyn Jones (born 1985), Welsh rugby union player David Wyn Jones (born 1950), British musicologist Enid Wyn Jones (1909–1967), Welsh nurse Ieuan Wyn Jones (born 1949), Welsh politician Richard Wyn Jones (born 1966), Welsh political scientist

Ending:

> Wyn Jones Wyn Jones may refer to: Wyn Jones (colonial administrator) (1926–1993), British colonial administrator Wyn Jones (police officer) (born c. 1943), British police officer, Assistant Commissioner of the Metropolitan Police Wyn Jones (rugby union) (born 1992), Welsh rugby union player See also Alun Wyn Jones (born 1985), Welsh rugby union player David Wyn Jones (born 1950), British musicologist Enid Wyn Jones (1909–1967), Welsh nurse Ieuan Wyn Jones (born 1949), Welsh politician Richard Wyn Jones (born 1966), Welsh political scientist

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Alachua County, Florida (`rec_3eb8a25f5fc088a7e9f20ba6`)

Band/weight: B0 / 0.25; caps: ['general_linewise_list', 'general_sensitive_context_review']; flags: ['human_sensitive_context_reviewed', 'linewise_list']

Chunk: 3/3; end boundary: paragraph; characters: 2,466; paragraphs: 13; alpha: 0.817; repeated trigrams: 0.049

Beginning:

> Alachua County, Florida On August 9, 2021, a prison inmate, Erica Thompson, gave birth while being held in the county jail. Her baby died. Despite the mother's screams, jail staff did not provide or call for medical assistance. An investigation held that law enforcement did not violate any law or policy. Landfills Alachua County is the site of five closed landfills—Southwest Landfill, Southeast Landfill, Northwest Landfill, Northeast Landfill, and Northeast Auxiliary Landfill. Since 1999, all solid waste from Alachua County has been hauled to the New River Solid Waste Facility in Raiford, in neighboring Union County. Communities Unincorporated communities Arredondo Bland Campville Cross Cre…

Ending:

> …ville. Spring Grove was the second county seat of Alachua County, after Newnansville was included in the newly created Columbia County, until Newnansville was returned to Alachua County and restored as the county seat. It was abandoned sometime in the middle of the 19th century. See also Alachua County Library District Florida State Parks in Alachua County National Register of Historic Places listings in Alachua County, Florida List of counties in Florida Notes External links Alachua County 1824 establishments in Florida Territory Florida placenames of Native American origin Charter counties in Florida Gainesville metropolitan area, Florida North Florida Populated places established in 1824

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Khaliji (music) (`rec_f02fc3ba0c4cf17c06b9a549`)

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

#### SV Arminen (`rec_1c7387b59680e5324ea3fef0`)

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

#### Clavulina rugosa (`rec_06b4e9b7e122abaef4b6000a`)

Band/weight: B2 / 1.0; caps: none; flags: none

Chunk: 1/1; end boundary: paragraph; characters: 400; paragraphs: 6; alpha: 0.790; repeated trigrams: 0.000

Beginning:

> Clavulina rugosa Clavulina rugosa, commonly known as the wrinkled coral fungus, is a species of coral fungus in the family Clavulinaceae. It is edible. Taxonomy The species was originally described as Clavaria rugosa by Jean Bulliard in 1790. It was transferred to Clavulina by Joseph Schröter in 1888. References External links Edible fungi Fungi described in 1790 Fungi of North America rugosa

Ending:

> Clavulina rugosa Clavulina rugosa, commonly known as the wrinkled coral fungus, is a species of coral fungus in the family Clavulinaceae. It is edible. Taxonomy The species was originally described as Clavaria rugosa by Jean Bulliard in 1790. It was transferred to Clavulina by Joseph Schröter in 1888. References External links Edible fungi Fungi described in 1790 Fungi of North America rugosa

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Main Beach Pavilion and Southport Surf Lifesaving Club (`rec_0706b14187a9895572541f41`)

Band/weight: B2 / 1.0; caps: none; flags: none

Chunk: 3/3; end boundary: paragraph; characters: 538; paragraphs: 6; alpha: 0.840; repeated trigrams: 0.043

Beginning:

> Main Beach Pavilion and Southport Surf Lifesaving Club Surf lifesaving Surf Life Saving Australia List of Australian surf lifesaving clubs References Attribution External links Queensland Heritage Register Southport, Queensland Tourist infrastructure in Queensland Main Beach, Queensland 1912 establishments in Australia Sports clubs and teams established in 1912 Surf Life Saving Australia clubs Sport on the Gold Coast, Queensland Articles incorporating text from the Queensland Heritage Register Gold Coast Local Heritage Register

Ending:

> Main Beach Pavilion and Southport Surf Lifesaving Club Surf lifesaving Surf Life Saving Australia List of Australian surf lifesaving clubs References Attribution External links Queensland Heritage Register Southport, Queensland Tourist infrastructure in Queensland Main Beach, Queensland 1912 establishments in Australia Sports clubs and teams established in 1912 Surf Life Saving Australia clubs Sport on the Gold Coast, Queensland Articles incorporating text from the Queensland Heritage Register Gold Coast Local Heritage Register

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Palinorsa raptans (`rec_cff2af48aa5a6b7546a9a36d`)

Band/weight: B2 / 1.0; caps: none; flags: none

Chunk: 1/1; end boundary: paragraph; characters: 714; paragraphs: 5; alpha: 0.794; repeated trigrams: 0.008

Beginning:

> Palinorsa raptans Palinorsa raptans is a moth in the family Depressariidae. It was described by Edward Meyrick in 1920. It is found in Peru and Brazil (Amazonas). The wingspan is about 28 mm. The forewings are dark brown, with violet iridescence and a suffused dark fuscous median longitudinal streak from the base to the apex, and one along the fold to a spot beneath the middle of the wing followed by a white dot, between these a band of greyish-violet suffusion extends to the termen. There is a transverse suffused dark fuscous spot from the upper margin of the median streak at three-fifths. The hindwings are grey, paler and subhyaline towards the base. References Moths described in 1920 Dep…

Ending:

> …aptans Palinorsa raptans is a moth in the family Depressariidae. It was described by Edward Meyrick in 1920. It is found in Peru and Brazil (Amazonas). The wingspan is about 28 mm. The forewings are dark brown, with violet iridescence and a suffused dark fuscous median longitudinal streak from the base to the apex, and one along the fold to a spot beneath the middle of the wing followed by a white dot, between these a band of greyish-violet suffusion extends to the termen. There is a transverse suffused dark fuscous spot from the upper margin of the median streak at three-fifths. The hindwings are grey, paler and subhyaline towards the base. References Moths described in 1920 Depressariinae

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### The Podium West Tower (`rec_1b4f19020a9a7ddc3a025d57`)

Band/weight: B2 / 1.0; caps: none; flags: none

Chunk: 1/1; end boundary: paragraph; characters: 930; paragraphs: 6; alpha: 0.791; repeated trigrams: 0.056

Beginning:

> The Podium West Tower The Podium West Tower is an 48-storey office skyscraper in Mandaluyong, Metro Manila, Philippines. It is part of The Podium mixed-used development, a project which was started in 2002. At its base, occupying the first five levels of the building, is The Podium shopping mall. The Podium shopping mall opened in 2002 but construction of The Podium West Tower would begin years later in 2015. The building topped-out on September 27, 2018 and overall construction of the tower was finished in May 2019. Prior to its completion, the US Green Building Council has given the building LEED Gold Mark certification. The Building and Construction Authority of Singapore also gave the b…

Ending:

> …ing the first five levels of the building, is The Podium shopping mall. The Podium shopping mall opened in 2002 but construction of The Podium West Tower would begin years later in 2015. The building topped-out on September 27, 2018 and overall construction of the tower was finished in May 2019. Prior to its completion, the US Green Building Council has given the building LEED Gold Mark certification. The Building and Construction Authority of Singapore also gave the building provisional Green Mark Gold Award. References Skyscrapers in Metro Manila Office buildings completed in 2019 Buildings and structures in Mandaluyong Arquitectonica buildings 21st-century architecture in the Philippines

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Women in warfare (1500–1699) (`rec_bb48103c30162b7c88a36630`)

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

#### 1995 Algerian presidential election (`rec_fde749ccde74975b87c0882e`)

Band/weight: B3 / 1.15; caps: none; flags: none

Chunk: 1/1; end boundary: paragraph; characters: 1,200; paragraphs: 7; alpha: 0.798; repeated trigrams: 0.038

Beginning:

> 1995 Algerian presidential election Presidential elections were held in Algeria on 16 November 1995, in the midst of the Algerian Civil War. The result was a victory for Liamine Zeroual, head of the High Council of State at the time, who won 61% of the vote. The Armed Islamic Group of Algeria threatened to kill anyone who voted, with the slogan "one vote, one bullet", but official voter turnout was 74.9%. Candidates Liamine Zeroual, independent Mahfoud Nahnah, candidate of the Islamist Movement of Society for Peace (MSP) Said Sadi, candidate of the secularist Rally for Culture and Democracy Noureddine Boukrouh, candidate of the Party of Algerian Renewal (PRA) Conduct Delegations of observer…

Ending:

> …ment of Society for Peace (MSP) Said Sadi, candidate of the secularist Rally for Culture and Democracy Noureddine Boukrouh, candidate of the Party of Algerian Renewal (PRA) Conduct Delegations of observers came from the Arab League, the African Union, and the United Nations, and reported no major problems. The Armed Islamic Group had threatened to kill voters, but the elections passed with few attacks. Voter turnout was high, despite the three largest parties of the 1991 parliamentary elections (the Islamic Salvation Front, National Liberation Front and Socialist Forces Front) calling for a boycott. Results References Algerian Civil War Presidential elections in Algeria Presidential Algeria

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Christmas in Fallujah (`rec_9e8514a19cb8a6d575119ad9`)

Band/weight: B3 / 1.15; caps: none; flags: none

Chunk: 1/1; end boundary: paragraph; characters: 1,660; paragraphs: 8; alpha: 0.783; repeated trigrams: 0.036

Beginning:

> Christmas in Fallujah "Christmas in Fallujah" is a single written by Billy Joel and performed by Cass Dillon. A couple of weeks after they recorded it in a studio, Billy Joel introduced Cass Dillon on stage, in Chicago, for a first live performance of the song. It is also Billy Joel's second new song of original material with lyrics he had written since 1993's River of Dreams. The single was released on December 4, 2007 exclusively from the iTunes Store and was included on Dillon's EP A Good Thing Never Dies (iTunes download). The proceeds from this single were donated to Homes for Our Troops, a nonprofit organization that builds specially adapted homes for American service members returnin…

Ending:

> …as in Fallujah" live in Australia in November 2008, marking the first time he sang the lyrics to the song instead of Dillon. On December 11, 2008, Joel announced that a new recording of the song that day at Sydney's Acer Arena concert would be released as a download and CD single in honor of the American and Australian soldiers serving in the Middle East. This is the only official recording of Joel singing "Christmas in Fallujah" that is available. Charts References External links Homes for Our Troops (official site) Songs of the Iraq War 2007 singles 2008 singles Billy Joel songs American Christmas songs Christmas charity singles Fallujah Live singles Songs written by Billy Joel 2007 songs

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Donbassaero (`rec_c55ff518016263f0ffcc81bc`)

Band/weight: B3 / 1.15; caps: none; flags: none

Chunk: 1/1; end boundary: paragraph; characters: 2,178; paragraphs: 13; alpha: 0.814; repeated trigrams: 0.014

Beginning:

> Donbassaero Donbassaero (, Russian: Донбассаэро) was an airline with its head office on the property of Donetsk International Airport in Donetsk, Ukraine. It operated domestic and international scheduled services. Its main bases were Donetsk International Airport and Boryspil International Airport in Kyiv. The main shareholder of the company was PrivatBank, controlled by Ihor Kolomoyskyi. History The airline was founded in 1993 as Donetsk State Airline, then re-organized and re-branded as Donbassaero in 2003. Their website was launched in July 2005 and their online booking system started in November of the same year. Since 25 March 2012, as a result of the Anti-monopoly committee of Ukraine…

Ending:

> …l Airport Syria Aleppo - Aleppo International Airport Turkey Istanbul - Atatürk International Airport United Arab Emirates Dubai - Dubai International Airport Ukraine Donetsk - Sergey Prokofiev International Airport hub Kharkiv - Kharkiv International Airport Kyiv - Boryspil International Airport hub Odesa - Odesa International Airport Fleet The Donbassaero fleet included the following aircraft (as of December 2012): References External links Official website Official website Donbassaero Fleet Photos Defunct airlines of Ukraine Airlines established in 2003 Airlines disestablished in 2013 Companies based in Donetsk Privat Group 2003 establishments in Ukraine 2013 disestablishments in Ukraine

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Geobacillus stearothermophilus (`rec_4881f16d445459e3cdae2208`)

Band/weight: B3 / 1.15; caps: none; flags: none

Chunk: 1/1; end boundary: paragraph; characters: 2,955; paragraphs: 12; alpha: 0.817; repeated trigrams: 0.016

Beginning:

> Geobacillus stearothermophilus Geobacillus stearothermophilus (previously Bacillus stearothermophilus) is a rod-shaped, Gram-positive bacterium and a member of the phylum Bacillota. The bacterium is a thermophile and is widely distributed in soil, hot springs, ocean sediment, and is a cause of spoilage in food products. It will grow within a temperature range of 30 to 75 °C. Some strains are capable of oxidizing carbon monoxide aerobically. It is commonly used as a challenge organism for sterilization validation studies and periodic check of sterilization cycles. The biological indicator contains spores of the organism on filter paper inside a vial. After sterilizing, the cap is closed, an…

Ending:

> …e group II intron reverse transcriptase (TGIRT), GsI-IIC-MRF, from G. stearothermophilus was found to retain activity up to 70 °C and to exhibit high processivity and a low error rate. These properties make this enzyme useful for reverse transcribing long and/or highly structured RNA molecules. A method for determining RNA secondary structure, DMS-MaPseq, uses this enzyme because it converts normal RNA to DNA accurately but introduces mutations at unpaired bases that have been methylated by dimethyl sulfate, and the mutations can be identified via sequencing. References External links Type strain of Geobacillus stearothermophilus at BacDive - the Bacterial Diversity Metadatabase Bacillaceae

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### The Adventures of Rocky and Bullwinkle and Friends (`rec_9a0aad97ba10d43020b2a3fa`)

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

#### Pennsylvania Department of Education (`rec_875355fe5fd7b9abf5c94540`)

Band/weight: B4 / 1.25; caps: none; flags: none

Chunk: 1/1; end boundary: paragraph; characters: 4,001; paragraphs: 14; alpha: 0.830; repeated trigrams: 0.078

Beginning:

> Pennsylvania Department of Education The Pennsylvania Department of Education is the executive department of the state charged with publicly funded preschool, K-12 and adult educational budgeting, management and guidelines. As the state education agency, its activities are directed by the governor appointed Pennsylvania's Secretary of Education. The agency is headquartered at 333 Market Street in Harrisburg. The Pennsylvania Department of Education oversees 500 public school districts of Pennsylvania, over 170 public charter schools (2019), Career and Technology Centers/Vocational Technical schools, 29 Intermediate Units, the education of youth in State Juvenile Correctional Institutions, a…

Ending:

> …State Board of Education Professional Standards and Practices Commission Office of Food and Nutrition Programs Special Education Advisory Panel State Boards of Private Schools Power Library Power Library is the online portal to Pennsylvania libraries, a service of the Office of Commonwealth Libraries, Pennsylvania Department of Education. Secretaries of Education See also List of Pennsylvania state agencies State education agency References External links Official website 1837 establishments in Pennsylvania Educational administration Government agencies established in 1837 Education, Department of Department State agencies of Pennsylvania State departments of education of the United States

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Thai labour law (`rec_3118eee1f14d4676e706052a`)

Band/weight: B4 / 1.25; caps: none; flags: none

Chunk: 1/1; end boundary: paragraph; characters: 5,595; paragraphs: 12; alpha: 0.794; repeated trigrams: 0.014

Beginning:

> Thai labour law The labour law of Thailand takes place under the framework of several acts of parliament and decrees, primarily the Labour Protection Act, B.E. 2541 (1998), and is mainly governed by the Ministry of Labour. Most of the legal framework was developed during the mid-to-late twentieth century, as Thailand's economy saw rapid expansion beginning in the Cold War period. While the law protects workers' rights of association and organization for collective bargaining, and allows workers to form unions, in practice the protections are inadequate, leading to a generally weak union system. The laws also only protect workers in the formal labour sector, and often don't reach Thailand's…

Ending:

> …. Foreign labour cap On 1 July 2018 a new labour law will go into effect, capping the number of foreign employees at businesses operating in Thailand. The move was taken to ensure Thais are not forced out of the labour market. Passed by the National Legislative Assembly in April 2018, the new law will restrict the number of foreign employees to a maximum of 20 percent of workforce in the industrial and services sectors. The law is opposed by business operators, especially those from small and medium-sized enterprises. (SMEs). The law will impact their hiring of low-cost migrant labour. See also Human trafficking in Thailand References Labour in Thailand Law of Thailand Labour law by country

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Sri Lankan cricket team in Pakistan in 2019–20 (`rec_cebd70f6102fa3d45ce42195`)

Band/weight: B4 / 1.25; caps: none; flags: none

Chunk: 1/2; end boundary: paragraph; characters: 7,298; paragraphs: 13; alpha: 0.777; repeated trigrams: 0.102

Beginning:

> Sri Lankan cricket team in Pakistan in 2019–20 The Sri Lankan cricket team toured Pakistan in September and October 2019 to play three One Day Internationals (ODIs) and three Twenty20 International (T20I) matches against the Pakistan cricket team. The tour originally had two Test matches scheduled to take place, but these were moved to December 2019. Sri Lanka last played a match in Pakistan in October 2017, when the third T20I took place at the Gaddafi Stadium in Lahore. Pakistan won the ODI series 2–0, after the first match was washed out, and Sri Lanka won the T20I series 3–0. Several players in Sri Lanka's squad opted not to travel for the series, with Lahiru Thirimanne and Dasun Shanak…

Ending:

> …world", with the hopes other teams would tour Pakistan again. Test matches In the first Test match, Sri Lanka won the toss and elected to bat. However, most of the match was affected by the weather, with rain and bad light impacting on the Test. Only 18.2 overs were bowled on day two, 5.2 overs on day three, and no play at all was possible on day four. Early on the fifth and final day, Sri Lanka declared their first innings, after Dhananjaya de Silva had scored his century. Pakistan's Abid Ali and Babar Azam both scored unbeaten hundreds, before the teams shook hands, with the match finishing as a draw. Abid Ali became the first male cricketer to score a century on his Test and ODI debuts.

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### A Klingon Christmas Carol (`rec_76868b56fb2fd2d2da65884d`)

Band/weight: B4 / 1.25; caps: none; flags: none

Chunk: 1/2; end boundary: paragraph; characters: 7,752; paragraphs: 33; alpha: 0.773; repeated trigrams: 0.082

Beginning:

> A Klingon Christmas Carol A Klingon Christmas Carol is the first play to be performed entirely in Klingon, a constructed language first appearing in the Star Trek media franchise. The play is based on the Charles Dickens 1843 novella, A Christmas Carol. A Klingon Christmas Carol is the Charles Dickens classic tale of ghosts and redemption, adapted to reflect the Klingon values of courage and honor, and then translated into Klingon, performed with English supertitles. Originally created as a fundraiser for Commedia Beauregard theatre company, it was written in 2007 by Christopher Kidder-Mostrom and Sasha Warren and was originally translated by Laura Thurston, Bill Hedrick and Christopher Kid…

Ending:

> …Huch qoy'wI' 1 and 2 (Charity Men) marlI' (Jacob Marley) ben qeylIS qa' (Ghost of Kahless Past) Qe'pa (a youth, Scrooge's schoolmate) Qob (a youth, Scrooge's schoolmate) SQujsa' Up (Young Scrooge) Van (Fannie, Scrooge's sister) veSIwIq (Fezziwig, Scrooge's employer) wIlqInS (Dick Wilkins, Scrooge's co-worker/Fezziwig's employee) bel (Belle) DaHjaj qeylIS qa' (Ghost of Kahless Present) 'emlI' (Mrs. Cratchit) marDa' (Martha Cratchit) tImHom (Tiny Tim) mara' (Mrs. Fred) meb (Guest) 1, 2 and 3 quvHa'ghach (Corruption) SaHHa'ghach (Apathy) pIq qeylIS qa' (Ghost of Kahless Yet-To-Come) SuvwI' (Warrior) 1, 2 and 3 Suy (Weapon Merchant) ngevwI' (Seller) loDHom (Boy) Production design Design history

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Eirik Kristoffersen (`rec_eb76ac9a1f5928465a3df807`)

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

#### Squamura maculata (`rec_122411c5ea64104f74d591ae`)

Band/weight: B0 / 0.25; caps: ['general_short']; flags: ['short_document']

Chunk: 1/1; end boundary: paragraph; characters: 300; paragraphs: 5; alpha: 0.813; repeated trigrams: 0.000

Beginning:

> Squamura maculata Squamura maculata is a moth in the family Cossidae. It is found on Sumatra, Borneo, Java and possibly in Cambodia. The habitat consists of lowland and lower montane forests. References Natural History Museum Lepidoptera generic names catalog Metarbelinae Moths described in 1890

Ending:

> Squamura maculata Squamura maculata is a moth in the family Cossidae. It is found on Sumatra, Borneo, Java and possibly in Cambodia. The habitat consists of lowland and lower montane forests. References Natural History Museum Lepidoptera generic names catalog Metarbelinae Moths described in 1890

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Puli Kiadeh (`rec_3d0e0788ba803c92c567890b`)

Band/weight: B0 / 0.25; caps: ['general_short']; flags: ['short_document']

Chunk: 1/1; end boundary: paragraph; characters: 322; paragraphs: 4; alpha: 0.767; repeated trigrams: 0.000

Beginning:

> Puli Kiadeh Puli Kiadeh (, also Romanized as Pūlī Kīādeh; also known as Polīkīādeh and Pūl Kīādeh) is a village in Harazpey-ye Jonubi Rural District, in the Central District of Amol County, Mazandaran Province, Iran. At the 2006 census, its population was 321, in 87 families. References Populated places in Amol County

Ending:

> Puli Kiadeh Puli Kiadeh (, also Romanized as Pūlī Kīādeh; also known as Polīkīādeh and Pūl Kīādeh) is a village in Harazpey-ye Jonubi Rural District, in the Central District of Amol County, Mazandaran Province, Iran. At the 2006 census, its population was 321, in 87 families. References Populated places in Amol County

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Ministry of Defence and Veterans Affairs (South Sudan) (`rec_e17aa47adaadd2ce2625fc68`)

Band/weight: B0 / 0.25; caps: ['general_short']; flags: ['short_document']

Chunk: 1/1; end boundary: paragraph; characters: 349; paragraphs: 4; alpha: 0.802; repeated trigrams: 0.259

Beginning:

> Ministry of Defence and Veterans Affairs (South Sudan) The Ministry of Defence and Veterans Affairs is a ministry of the Government of South Sudan. The incumbent minister is Chol Thon Balok who assumed the post on 29 March 2023. References Defence and Veterans Affairs South Sudan, Defence and Veterans Affairs South Sudan Military of South Sudan

Ending:

> Ministry of Defence and Veterans Affairs (South Sudan) The Ministry of Defence and Veterans Affairs is a ministry of the Government of South Sudan. The incumbent minister is Chol Thon Balok who assumed the post on 29 March 2023. References Defence and Veterans Affairs South Sudan, Defence and Veterans Affairs South Sudan Military of South Sudan

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Everett Whittingham (`rec_4f3e4d105ce5f5cf2b3916fd`)

Band/weight: B0 / 0.25; caps: ['general_short']; flags: ['short_document']

Chunk: 1/1; end boundary: paragraph; characters: 373; paragraphs: 6; alpha: 0.783; repeated trigrams: 0.000

Beginning:

> Everett Whittingham Everett Whittingham (born 25 February 1954) is a Jamaican cricketer. He played in one first-class and three List A matches for the Jamaican cricket team from 1980 to 1985. See also List of Jamaican representative cricketers References External links 1954 births Living people Jamaican cricketers Jamaica cricketers Cricketers from Kingston, Jamaica

Ending:

> Everett Whittingham Everett Whittingham (born 25 February 1954) is a Jamaican cricketer. He played in one first-class and three List A matches for the Jamaican cricket team from 1980 to 1985. See also List of Jamaican representative cricketers References External links 1954 births Living people Jamaican cricketers Jamaica cricketers Cricketers from Kingston, Jamaica

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Dehgeh-ye Shah Mansuri (`rec_e6e195642ce0c0606114ce55`)

Band/weight: B0 / 0.25; caps: ['general_short']; flags: ['short_document']

Chunk: 1/1; end boundary: paragraph; characters: 399; paragraphs: 4; alpha: 0.789; repeated trigrams: 0.068

Beginning:

> Dehgeh-ye Shah Mansuri Dehgeh-ye Shah Mansuri (, also Romanized as Dehgeh-ye Shāh Manşūrī) is a village in Doab Rural District, Bazoft District, Kuhrang County, Chaharmahal and Bakhtiari Province, Iran. At the 2006 census, its population was 99, in 18 families. The village is populated by Lurs. References Populated places in Kuhrang County Luri settlements in Chaharmahal and Bakhtiari Province

Ending:

> Dehgeh-ye Shah Mansuri Dehgeh-ye Shah Mansuri (, also Romanized as Dehgeh-ye Shāh Manşūrī) is a village in Doab Rural District, Bazoft District, Kuhrang County, Chaharmahal and Bakhtiari Province, Iran. At the 2006 census, its population was 99, in 18 families. The village is populated by Lurs. References Populated places in Kuhrang County Luri settlements in Chaharmahal and Bakhtiari Province

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

### cap_general_disambiguation

#### David Preece (`rec_0c542d6eb56aaf8f67212d30`)

Band/weight: B0 / 0.25; caps: ['general_short', 'general_disambiguation']; flags: ['disambiguation_page', 'short_document']

Chunk: 1/1; end boundary: paragraph; characters: 302; paragraphs: 2; alpha: 0.745; repeated trigrams: 0.048

Beginning:

> David Preece David Preece may refer to: David Preece (footballer, born 1963) (1963–2007), played for Walsall, Luton Town, Derby County and Cambridge United David Preece (footballer, born 1976), played for Darlington, Aberdeen and Silkeborg IF David Preece (racing driver), former British racing driver

Ending:

> David Preece David Preece may refer to: David Preece (footballer, born 1963) (1963–2007), played for Walsall, Luton Town, Derby County and Cambridge United David Preece (footballer, born 1976), played for Darlington, Aberdeen and Silkeborg IF David Preece (racing driver), former British racing driver

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Emily Martin (`rec_0097ed6b6baa9e7aa74da8c1`)

Band/weight: B0 / 0.25; caps: ['general_short', 'general_disambiguation']; flags: ['disambiguation_page', 'short_document']

Chunk: 1/1; end boundary: paragraph; characters: 378; paragraphs: 3; alpha: 0.762; repeated trigrams: 0.000

Beginning:

> Emily Martin Emily Martin may refer to: Emily Martin (1884–1962), aka Emily Dutton, South Australian musician and socialite Emily Martin (anthropologist) (born 1944), sinologist, anthropologist, and feminist Emily Martin (rower) (born 1979), Australian rower Emily Martin (diver), British diver Emily Winfield Martin, American artist and author-illustrator of children's books

Ending:

> Emily Martin Emily Martin may refer to: Emily Martin (1884–1962), aka Emily Dutton, South Australian musician and socialite Emily Martin (anthropologist) (born 1944), sinologist, anthropologist, and feminist Emily Martin (rower) (born 1979), Australian rower Emily Martin (diver), British diver Emily Winfield Martin, American artist and author-illustrator of children's books

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### One Mississippi (`rec_362bcf3b9b56f5d1a06fdf5d`)

Band/weight: B0 / 0.25; caps: ['general_disambiguation']; flags: ['disambiguation_page']

Chunk: 1/1; end boundary: paragraph; characters: 539; paragraphs: 3; alpha: 0.722; repeated trigrams: 0.136

Beginning:

> One Mississippi One Mississippi may refer to: One Mississippi (Brendan Benson album), 1996 One Mississippi (J Church album), 2000 One Mississippi (TV series), a 2016 American television series "One Mississippi", a song on the 2003 album Jillbilly by Jill King "One Mississippi", a song on the 2013 album Bring You Back by Brett Eldredge "One Mississippi", a song on the 2017 album So Good by Zara Larsson "One Mississippi", a song on the 2020 album My Mississippi Reunion by Steve Azar "One Mississippi" (song), a 2021 song by Kane Brown

Ending:

> One Mississippi One Mississippi may refer to: One Mississippi (Brendan Benson album), 1996 One Mississippi (J Church album), 2000 One Mississippi (TV series), a 2016 American television series "One Mississippi", a song on the 2003 album Jillbilly by Jill King "One Mississippi", a song on the 2013 album Bring You Back by Brett Eldredge "One Mississippi", a song on the 2017 album So Good by Zara Larsson "One Mississippi", a song on the 2020 album My Mississippi Reunion by Steve Azar "One Mississippi" (song), a 2021 song by Kane Brown

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Nandu (`rec_8e503a5bf1123e08280fe758`)

Band/weight: B0 / 0.25; caps: ['general_disambiguation']; flags: ['disambiguation_page']

Chunk: 1/1; end boundary: paragraph; characters: 830; paragraphs: 6; alpha: 0.748; repeated trigrams: 0.052

Beginning:

> Nandu Nandu may refer to: Places Chengdu, a city in Sichuan, China, known as (Southern Capital or Nandu) during the early Tang dynasty Jiangling County, a city in Hubei, China, formerly known as (Southern Capital or Nandu) during the later Tang dynasty Nandu River, Hainan province, China Other uses Ñandú, a native South American name for any of three species of Rhea. Nandu (film), a 1981 Tamil film Ñandú (vehicle), a 1940s all-terrain vehicle military vehicle Southern Metropolis Daily, often shortened to Nandu (南都) One of the Argentine Air Force flights that attacked the British fleet in the Battle of San Carlos, during the Falklands War, 1982 People with the given name Nandu Bhende (c. 195…

Ending:

> …dynasty Jiangling County, a city in Hubei, China, formerly known as (Southern Capital or Nandu) during the later Tang dynasty Nandu River, Hainan province, China Other uses Ñandú, a native South American name for any of three species of Rhea. Nandu (film), a 1981 Tamil film Ñandú (vehicle), a 1940s all-terrain vehicle military vehicle Southern Metropolis Daily, often shortened to Nandu (南都) One of the Argentine Air Force flights that attacked the British fleet in the Battle of San Carlos, during the Falklands War, 1982 People with the given name Nandu Bhende (c. 1955–2014), Indian singer Nandu M. Natekar (1933–2021), Indian badminton player See also Nandhu (born 1965), Malayalam film actor

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Old Post Office (`rec_ce06fe555e95c16ebadfa33b`)

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

#### Futsal at the 2007 Asian Indoor Games (`rec_2954393965835be7dc2ea3e0`)

Band/weight: B0 / 0.25; caps: ['general_structured_low_prose']; flags: ['duplicate_lines']

Chunk: 1/1; end boundary: paragraph; characters: 650; paragraphs: 31; alpha: 0.754; repeated trigrams: 0.165

Beginning:

> Futsal at the 2007 Asian Indoor Games Futsal at the 2007 Asian Indoor Games was held in Macau, China from 26 October to 3 November 2007. Medalists Medal table Results Men Preliminary Group A Group B Group C Group D Kuwait was disqualified from the tournament on 29 October after Kuwait Football Association was suspended by FIFA. Knockout round Quarterfinals Semifinals Bronze medal match Gold medal match Goalscorers Women Preliminary Group A Group B Placing Knockout round Semifinals Bronze medal match Gold medal match Goalscorers References RSSSF 2007 Asian Indoor Games events Indoor Games 2007 2007 Futsal in Macau

Ending:

> Futsal at the 2007 Asian Indoor Games Futsal at the 2007 Asian Indoor Games was held in Macau, China from 26 October to 3 November 2007. Medalists Medal table Results Men Preliminary Group A Group B Group C Group D Kuwait was disqualified from the tournament on 29 October after Kuwait Football Association was suspended by FIFA. Knockout round Quarterfinals Semifinals Bronze medal match Gold medal match Goalscorers Women Preliminary Group A Group B Placing Knockout round Semifinals Bronze medal match Gold medal match Goalscorers References RSSSF 2007 Asian Indoor Games events Indoor Games 2007 2007 Futsal in Macau

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Russian Professional Basketball League Awards (`rec_cea13290c447a298a387847a`)

Band/weight: B0 / 0.25; caps: ['general_structured_low_prose']; flags: ['duplicate_lines']

Chunk: 1/1; end boundary: paragraph; characters: 1,606; paragraphs: 15; alpha: 0.814; repeated trigrams: 0.230

Beginning:

> Russian Professional Basketball League Awards The Russian Professional Basketball League Awards were the awards that were given out by the former top-tier level professional basketball league in Russia, the Russian Professional Basketball League (RPBL). PBL Awards Russian Professional Basketball League (PBL) 2010–11 season awards PBL Regular Season MVP Maciej Lampe (UNICS Kazan) PBL Playoffs MVP Victor Khryapa (CSKA Moscow) PBL All-Symbolic Team PBL First Symbolic Team Patrick Beverley (Spartak St. Petersburg) Keith Langford (Khimki Moscow Region) Henry Domercant (Spartak St. Petersburg) Sergei Monia (Khimki Moscow Region) Maciej Lampe (UNICS Kazan) PBL Second Symbolic Team Marcus Williams…

Ending:

> …L) 2011–12 season awards PBL Regular Season MVP Davon Jefferson (Triumph Lyubertsy) PBL Playoffs MVP Alexey Shved (CSKA Moscow) PBL All-Symbolic Team PBL First Symbolic Team Patrick Beverley (Spartak St. Petersburg) Zoran Planinić (Khimki Moscow Region) Davon Jefferson (Triumph Lyubertsy) Andrei Kirilenko (CSKA Moscow) Jeremiah Massey (Lokomotiv Kuban) PBL Second Symbolic Team Torey Thomas (Spartak Primorye) Vitaly Fridzon (Khimki Moscow Region) Sergey Karasev (Triumph Lyubertsy) Victor Khryapa (CSKA Moscow) Vladimir Veremeenko (UNICS Kazan) See also Russian Gold Basket Awards References External links Russian Professional Basketball League official website awards European basketball awards

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### 1933 Campeonato Carioca (`rec_db549d25f278ef507ce108b3`)

Band/weight: B0 / 0.25; caps: ['general_structured_low_prose']; flags: ['duplicate_lines']

Chunk: 1/1; end boundary: paragraph; characters: 2,478; paragraphs: 16; alpha: 0.796; repeated trigrams: 0.115

Beginning:

> 1933 Campeonato Carioca In the 1933 season of the Campeonato Carioca, two championships were disputed, each by a different league. AMEA Championship After the 1932 championship, talks began among the seven main clubs of the AMEA league to discuss whether to adopt professionalism, like APEA in São Paulo had done before, or not. However, after the league's statue was first drafted, only América, Bangu and Fluminense accepted it, although they were joined by Vasco da Gama, which reversed its previous position on that matter. The four teams were consequently expelled from AMEA, which was resolved to remain amateur. Later on, Bonsucesso joined them, and CBD took a stance against professionalism,…

Ending:

> …championship for the 6th time. no teams were relegated. Participating teams System The tournament would be disputed in a double round-robin format, with the team with the most points winning the title. Championship LCF Championship The edition of the Campeonato Carioca organized by LCF (Liga Carioca de Football, or Carioca Football League) kicked off on May 7, 1933, and ended on November 15, 1933. Six teams participated. Bangu won the championship for the 1st time. no teams were relegated. Participating teams System The tournament would be disputed in a double round-robin format, with the team with the most points winning the title. Championship References Campeonato Carioca seasons Carioca

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Virginia State Route 143 (`rec_8a97bf877114665f39d8461d`)

Band/weight: B0 / 0.25; caps: ['general_structured_low_prose']; flags: ['duplicate_lines']

Chunk: 1/1; end boundary: paragraph; characters: 7,048; paragraphs: 14; alpha: 0.772; repeated trigrams: 0.124

Beginning:

> Virginia State Route 143 State Route 143 (SR 143) is a primary state highway in the U.S. state of Virginia. The state highway runs from Camp Peary near Williamsburg east to U.S. Route 258 (US 258) at Fort Monroe in Hampton. SR 143 is a major local thoroughfare on the Virginia Peninsula portion of the Hampton Roads metropolitan area. The state highway is named Merrimac Trail through the independent city of Williamsburg and adjacent portions of York County and James City County. SR 143 follows Jefferson Avenue through the city of Newport News from the Williamsburg area past Virginia Peninsula Regional Jail to near Downtown Newport News. The state highway, which mostly runs northwest–southeast…

Ending:

> …e with I-64 (Hampton Roads Beltway), which US 60 joins to cross Hampton Roads via the Hampton Roads Bridge-Tunnel to Norfolk. SR 143 continues southeast along two-lane County Street, turns southwest onto Libby Street for one block, then turns south on Mellen Street and intersects SR 169 (Mallory Street) within the Phoebus neighborhood. The state highway crosses the Mill Creek estuary as Ingalls Road and reaches its eastern terminus at its junction with US 258's (Mercury Boulevard) northern terminus at the entrance to Fort Monroe. Major intersections References External links Virginia Highways Project: VA 143 143 State Route 143 State Route 143 State Route 143 State Route 143 State Route 143

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### List of Solanales of South Africa (`rec_ff59dda81ebf0224f65ec240`)

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

#### William J. Haynes II (`rec_e3c0d27eb90200dae01adc2b`)

Band/weight: B0 / 0.25; caps: ['general_linewise_list', 'general_category_tail']; flags: ['category_tail', 'linewise_list', 'short_continuation_chunk']

Chunk: 3/3; end boundary: paragraph; characters: 306; paragraphs: 2; alpha: 0.824; repeated trigrams: 0.023

Beginning:

> William J. Haynes II 1958 births American lawyers Davidson College alumni General Counsels of the United States Army George W. Bush administration personnel Harvard Law School alumni Living people People associated with Jenner & Block People from Waco, Texas Texas Republicans Torture in the United States

Ending:

> William J. Haynes II 1958 births American lawyers Davidson College alumni General Counsels of the United States Army George W. Bush administration personnel Harvard Law School alumni Living people People associated with Jenner & Block People from Waco, Texas Texas Republicans Torture in the United States

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Dejan Mitrović (`rec_9ef9b887f53e6426bcc3811f`)

Band/weight: B0 / 0.25; caps: ['general_linewise_list']; flags: ['linewise_list']

Chunk: 1/1; end boundary: paragraph; characters: 1,282; paragraphs: 5; alpha: 0.805; repeated trigrams: 0.074

Beginning:

> Dejan Mitrović Dejan Mitrović (Serbian Cyrillic: Дејан Митровић; born February 2, 1973) is a retired Serbian football player. Career After playing with OFK Beograd still a youngster, he will move in 1992 to Belgium where he will spend most of his career. After playing initially with a minor club Nijlen, he moved in 1994 to FC Kapellen and will play afterwards for several other Belgian clubs, such as Belgian First Division clubs KVC Westerlo, R.E. Mouscron and, later on, with Royal Antwerp FC and lower league KFC Lille. In between, he will also have spells in Cyprus, with Anorthosis Famagusta and Anagennisi Dherynia, and in Portugal, with C.F. União from Madeira. External links Living people…

Ending:

> …northosis Famagusta and Anagennisi Dherynia, and in Portugal, with C.F. União from Madeira. External links Living people 1973 births Serbian men's footballers Serbian expatriate men's footballers OFK Beograd players K.V.C. Westerlo players Royal Excel Mouscron players Belgian Pro League players Challenger Pro League players Expatriate men's footballers in Belgium Anorthosis Famagusta FC players Anagennisi Deryneia FC players Expatriate men's footballers in Cyprus C.F. União players Expatriate men's footballers in Portugal Royal Antwerp F.C. players Men's association football midfielders Royal Cappellen F.C. players Cypriot First Division players Liga Portugal 2 players People from Obrenovac

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Euratom Treaty (`rec_9938a200a2a657dd4f42fd90`)

Band/weight: B0 / 0.25; caps: ['general_linewise_list']; flags: ['linewise_list']

Chunk: 1/1; end boundary: paragraph; characters: 2,586; paragraphs: 10; alpha: 0.813; repeated trigrams: 0.096

Beginning:

> Euratom Treaty The Euratom Treaty, officially the Treaty establishing the European Atomic Energy Community, established the European Atomic Energy Community. It was signed on 25 March 1957 at the same time as the Treaty establishing the European Economic Community (EEC Treaty). The Euratom Treaty is less well known because of the lower profile of the organisation that it founded. The EEC has evolved into what is now the European Union, but Euratom has remained much the same as it was in 1957 although it is governed by the institutions of the European Union. It was established with its own independent institutions, but the 1967 Merger Treaty merged the institutions of Euratom and the Europea…

Ending:

> …chnology treaties Treaties of Austria Treaties of Bulgaria Treaties of Belgium Treaties of Croatia Treaties of Cyprus Treaties of the Czech Republic Treaties of Denmark Treaties of Estonia Treaties of Finland Treaties of the French Fourth Republic Treaties of West Germany Treaties of Greece Treaties of Hungary Treaties of Ireland Treaties of Italy Treaties of Latvia Treaties of Lithuania Treaties of Luxembourg Treaties of Malta Treaties of the Netherlands Treaties of Poland Treaties of Portugal Treaties of Romania Treaties of Slovakia Treaties of Slovenia Treaties of Spain Treaties of Sweden Treaties of the United Kingdom Treaties extended to Åland March 1957 events in Europe Events in Rome

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### 2010 Ghana Movie Awards (`rec_d84e2c3e006d4c73acc5b47a`)

Band/weight: B0 / 0.25; caps: ['general_linewise_list']; flags: ['linewise_list']

Chunk: 1/1; end boundary: paragraph; characters: 4,884; paragraphs: 30; alpha: 0.759; repeated trigrams: 0.176

Beginning:

> 2010 Ghana Movie Awards The 2010 Ghana Movie Awards was the maiden edition of the ceremony to reward cinematic achievement in Ghana Film Industry. The event was held at the Golden Tulip Hotel, Accra on 25 December 2010. Sinking Sands, Juliet Ibrahim, Nadia Buari, Yvonne Okoro, Majid Michel, John Dumelo & Genevieve Nnaji were among the winners. Awards Best Actor in a Leading Role (English) Senanu Gbedawu (Check Mate) Majid Michel (The Beast) J.O.T Agyemany (I Sing of a Well) Prince Osei (Kiss Me If You Can) Eddie Nartey (Kiss Me If You Can) Van Vicker (Dna Test) Ruffy Samuel (Love & Lust) Best Actress in a Leading Role (English) Martha Ankomah (Kiss Me If You Can) Akorfa Edjeani Asiedu (I Si…

Ending:

> …ng Sands) Ramsey Nouah (Guilty Pleasures) Desmond Elliot (Guilty Pleasures) Uti Nwachukwu (Busting Out (film)) Best Actress - West Africa Collaboration Genevieve Nnaji (Silent Scandals) Nse Ikpe Etim (Guilty Pleasures (2009 film)) Tonto Dikeh (Love & Lust) Uche Jombo (Nollywood Hustlers) Omotola Jalade Ekeinde (Private Storm) Mercy Johnson (Shakira) Best Movie - African Collaboration Sinking Sands Guilty Pleasures (2009 film) Love & Lust Private Storm Bursting Out (film) Best Movie Score The Game (2010 film) Ama Ghana A Sting in a Tale Kiss Me If You Can 4 Play (film) Sinking Sands Favorite Actor Kofi Adjorlolo Favorite Actress Yvonne Nelson References Ghana Movie Awards Ghana 2010 in Ghana

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Khaliji (music) (`rec_f02fc3ba0c4cf17c06b9a549`)

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

#### Caning (`rec_6f908dd96f503ab796b29af1`)

Band/weight: B0 / 0.25; caps: ['general_category_tail']; flags: ['category_tail', 'short_continuation_chunk']

Chunk: 3/3; end boundary: paragraph; characters: 77; paragraphs: 2; alpha: 0.883; repeated trigrams: 0.000

Beginning:

> Caning BDSM activities Corporal punishments School punishments Whipping Pain

Ending:

> Caning BDSM activities Corporal punishments School punishments Whipping Pain

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Clinton Correctional Facility (`rec_7f7ceeff46829405d7e5a7cc`)

Band/weight: B0 / 0.25; caps: ['general_category_tail']; flags: ['category_tail', 'short_continuation_chunk']

Chunk: 2/2; end boundary: paragraph; characters: 190; paragraphs: 2; alpha: 0.795; repeated trigrams: 0.154

Beginning:

> Clinton Correctional Facility Buildings and structures in Clinton County, New York Capital punishment in New York (state) Prisons in New York (state) 1845 establishments in New York (state)

Ending:

> Clinton Correctional Facility Buildings and structures in Clinton County, New York Capital punishment in New York (state) Prisons in New York (state) 1845 establishments in New York (state)

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Thomas Goode (merchant) (`rec_26f1a778811cb2f48e2c3f52`)

Band/weight: B0 / 0.25; caps: ['general_category_tail']; flags: ['category_tail', 'short_continuation_chunk']

Chunk: 3/3; end boundary: paragraph; characters: 239; paragraphs: 3; alpha: 0.808; repeated trigrams: 0.000

Beginning:

> Thomas Goode (merchant) References 1816 births 1882 deaths People from Goolwa, South Australia Settlers of South Australia English emigrants to colonial Australia 19th-century Australian businesspeople 19th-century English businesspeople

Ending:

> Thomas Goode (merchant) References 1816 births 1882 deaths People from Goolwa, South Australia Settlers of South Australia English emigrants to colonial Australia 19th-century Australian businesspeople 19th-century English businesspeople

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Vatican City during World War II (`rec_73602811bc4b8c047fc92c67`)

Band/weight: B0 / 0.25; caps: ['general_linewise_list', 'general_category_tail']; flags: ['category_tail', 'linewise_list', 'short_continuation_chunk']

Chunk: 4/4; end boundary: paragraph; characters: 333; paragraphs: 2; alpha: 0.733; repeated trigrams: 0.153

Beginning:

> Vatican City during World War II Neutral states in World War II Pope Pius XII and World War II World War II national military histories Wars involving Vatican City History of the papacy 1939 in Vatican City 1940 in Vatican City 1941 in Vatican City 1942 in Vatican City 1943 in Vatican City 1944 in Vatican City 1945 in Vatican City

Ending:

> Vatican City during World War II Neutral states in World War II Pope Pius XII and World War II World War II national military histories Wars involving Vatican City History of the papacy 1939 in Vatican City 1940 in Vatican City 1941 in Vatican City 1942 in Vatican City 1943 in Vatican City 1944 in Vatican City 1945 in Vatican City

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Giovanni Hidalgo (`rec_dd26e1a2c99589b824807963`)

Band/weight: B0 / 0.25; caps: ['general_linewise_list', 'general_category_tail']; flags: ['category_tail', 'linewise_list', 'short_continuation_chunk']

Chunk: 2/2; end boundary: paragraph; characters: 392; paragraphs: 2; alpha: 0.860; repeated trigrams: 0.000

Beginning:

> Giovanni Hidalgo 1963 births Living people American percussionists American drummers Latin jazz percussionists Conga players Barril players Plenera players Puerto Rican educators Music educators musicians from San Juan, Puerto Rico Planet Drum members Djembe players Batá drummers Timbaleros Bongo players American marimbists Timpanists Tubular bells players Tambourine players Güiro players

Ending:

> Giovanni Hidalgo 1963 births Living people American percussionists American drummers Latin jazz percussionists Conga players Barril players Plenera players Puerto Rican educators Music educators musicians from San Juan, Puerto Rico Planet Drum members Djembe players Batá drummers Timbaleros Bongo players American marimbists Timpanists Tubular bells players Tambourine players Güiro players

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

### cap_general_sensitive_context_review

#### Alachua County, Florida (`rec_3eb8a25f5fc088a7e9f20ba6`)

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

#### List of Saxifragales of South Africa (`rec_1c8a71e1fb59c059cc3e226e`)

Band/weight: B0 / 0.25; caps: ['general_linewise_list']; flags: ['linewise_list']

Chunk: 2/4; end boundary: line; characters: 7,963; paragraphs: 2; alpha: 0.815; repeated trigrams: 0.349

Beginning:

> List of Saxifragales of South Africa Crassula alcicornis Schonland, endemic Crassula alpestris Thunb. indigenous Crassula alpestris Thunb. subsp. alpestris, endemic Crassula alpestris Thunb. subsp. massonii (Britten & Baker f.) Toelken, endemic Crassula alstonii Marloth, endemic Crassula ammophila Toelken, endemic Crassula aphylla Schonland & Baker f. endemic Crassula arborea Medik. accepted as Crassula arborescens (Mill.) Willd. subsp. arborescens, indigenous Crassula arborescens (Mill.) Willd. endemic Crassula arborescens (Mill.) Willd. subsp. arborescens, endemic Crassula arborescens (Mill.) Willd. subsp. undulatifolia Toelken, endemic Crassula argentea Thunb. accepted as Crassula ovata…

Ending:

> …s (Haw.) D.Dietr. subsp. hispida (Haw.) Toelken, endemic Crassula mesembryanthemoides (Haw.) D.Dietr. subsp. mesembryanthemoides, endemic Crassula minuta Toelken, endemic Crassula mollis Thunb. endemic Crassula montana Thunb. indigenous Crassula montana Thunb. subsp. montana, endemic Crassula montana Thunb. subsp. quadrangularis (Schonland) Toelken, endemic Crassula multicava Lem. indigenous Crassula multicava Lem. subsp. floribunda Friedrich ex Toelken, endemic Crassula multicava Lem. subsp. multicava, endemic Crassula multiceps Harv. endemic Crassula multiflora Schonland & Baker f. indigenous Crassula multiflora Schonland & Baker f. subsp. leucantha (Schonland & Baker f.) Toelken, endemic

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Anagyrus (`rec_28f54d8bc068fc424d7c2d40`)

Band/weight: B0 / 0.25; caps: ['general_linewise_list']; flags: ['linewise_list']

Chunk: 1/2; end boundary: line; characters: 7,985; paragraphs: 3; alpha: 0.705; repeated trigrams: 0.212

Beginning:

> Anagyrus Anagyrus is a large genus of parasitic wasps from the family Encyrtidae. Anagyrus is distributed throughout the world. A subgenus of Anagyrus is known as Nesoanagyrus (Beardsley 1969) Species There are at least 247 species in this genus which consists of: Anagyrus abatos (Noyes & Menezes, 2000) Anagyrus abdulrassouli (Myartseva, Sugonjaev & Trjapitzin, 1982) Anagyrus abyssinicus Compere, 1939 Anagyrus aceris Noyes & Hayat, 1994 Anagyrus aciculatus (Blanchard, 1940) Anagyrus adamsoni Timberlake, 1941 Anagyrus aega Noyes, 2000 Anagyrus aegyptiacus Moursi, 1948 Anagyrus agraensis Saraswat 1975 Anagyrus alami Hayat 1970 Anagyrus albatus Myartseva, 1982 Anagyrus aligarhensis Agarwal & A…

Ending:

> …Domenichini, 1953 Anagyrus semifulvus Girault, 1915 Anagyrus shahidi Hayat, 1979 Anagyrus siccus (Prinsloo & Annecke, 1976) Anagyrus similis (Girault 1915) Anagyrus sinensis Noyes & Hayat, 1994 Anagyrus sinope Noyes & Menezes 2000 Anagyrus smithi Doutt, 1952 Anagyrus sogdianus Sugonjaev, 1968 Anagyrus sophax Noyes & Menezes 2000 Anagyrus spaici (Hoffer, 1970) Anagyrus spica (Girault 1921) Anagyrus subalbipes Ishii, 1928 Anagyrus subflaviceps (Girault 1915) Anagyrus subnigricornis Ishii, 1928 Anagyrus subproximus (Silvestri, 1915) Anagyrus subtilis Noyes & Hayat, 1994 Anagyrus sucro Noyes, 2000 Anagyrus suia Noyes, 2000 Anagyrus surekhae Noyes & Menezes 2000 Anagyrus swezeyi Timberlake, 1919

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### First-wave feminism (`rec_338cb4073008d9bfc7380e66`)

Band/weight: B0 / 0.25; caps: ['general_linewise_list']; flags: ['linewise_list']

Chunk: 9/11; end boundary: line; characters: 7,948; paragraphs: 2; alpha: 0.800; repeated trigrams: 0.178

Beginning:

> First-wave feminism Argentina: A group of anarcha-feminist women, headed by Virginia Bolten, publish La Voz de la Mujer, one of the first feminist newspapers of Latin America. US, Idaho: Idaho granted women the right to vote. 1900 Western Australia: Western Australia granted women the right to vote. Belgium: Legal majority was granted to unmarried women. Egypt: A school for female teachers was founded in Cairo. France: Women were allowed to practice law. Korea: The post office profession was opened to women. Tunisia: The first public elementary school for girls was opened. Japan: The first women's university was opened. Baden, Germany: Universities opened to women. Sweden: Maternity leave w…

Ending:

> …ondon Federation of Suffragettes. 1913 Russia: In 1913 Russian women observed their first International Women's Day on the last Sunday in February. Following discussions, International Women's Day was transferred to 8 March and this day has remained the global date for International Women's Day ever since. US, Alaska: Alaska granted women the right to vote. Norway: Norway granted women the right to vote. Japan: Public universities opened to women. United Kingdom: The suffragette Emily Davison was killed by the King's horse at The Derby. United Kingdom: 50,000 women taking part in a pilgrimage organized by the National Union of Women's Suffrage Societies arrived in Hyde Park on July 26. 1914

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### List of moths of Australia (Cosmopterigidae) (`rec_40a3cccc50bcbe278884f6fa`)

Band/weight: B0 / 0.25; caps: ['general_linewise_list']; flags: ['linewise_list']

Chunk: 1/2; end boundary: line; characters: 7,968; paragraphs: 3; alpha: 0.717; repeated trigrams: 0.153

Beginning:

> List of moths of Australia (Cosmopterigidae) This is a list of the Australian species of the family Cosmopterigidae. It also acts as an index to the species articles and forms part of the full List of moths of Australia. Chrysopeleiinae Cholotis exodroma (Meyrick, 1897) Cholotis semnostola (Meyrick, 1897) Eumenodora encrypta Meyrick, 1906 Ithome lassula Hodges, 1962 Leptozestis anagrapta (Meyrick, 1897) Leptozestis antithetis (Meyrick, 1897) Leptozestis argoscia (Lower, 1904) Leptozestis autochroa (Meyrick, 1915) Leptozestis capnopora (Meyrick, 1897) Leptozestis cataspoda (Meyrick, 1897) Leptozestis charmosyna (Meyrick, 1921) Leptozestis crassipalpis (Turner, 1923) Leptozestis crebra (Meyri…

Ending:

> …Limnaecia stenosticha Turner, 1926 Limnaecia symplecta Turner, 1923 Limnaecia syntaracta Meyrick, 1897 Limnaecia tetraplanetis Meyrick, 1897 Limnaecia triplaneta Meyrick, 1921 Limnaecia trisema Meyrick, 1897 Limnaecia trissodesma (Meyrick, 1887) Limnaecia trixantha (Lower, 1920) Limnaecia xanthopelta Lower, 1903 Limnaecia xanthopis Meyrick, 1920 Limnaecia zonomacula Lower, 1908 Limnaecia zotica Meyrick, 1921 Macrobathra allocrana Turner, 1916 Macrobathra allophyla (Turner, 1944) Macrobathra alternatella (Walker, 1864) Macrobathra anacampta Meyrick, 1914 Macrobathra anemarcha Meyrick, 1886 Macrobathra anemodes Meyrick, 1886 Macrobathra aneurae Turner, 1932 Macrobathra aphristis Meyrick, 1889

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### List of Saxifragales of South Africa (`rec_40f3ae1b3a5a6a74bd2ba187`)

Band/weight: B0 / 0.25; caps: ['general_linewise_list']; flags: ['linewise_list']

Chunk: 1/4; end boundary: line; characters: 7,975; paragraphs: 15; alpha: 0.808; repeated trigrams: 0.168

Beginning:

> List of Saxifragales of South Africa Saxifragales (saxifrages) is an order of flowering plants (Angiosperms). They are an extremely diverse group of plants which include trees, shrubs, perennial herbs, succulent and aquatic plants. The degree of diversity in terms of vegetative and floral features makes it difficult to define common features that unify the order. In the Angiosperm Phylogeny Group classification system, the Saxifragales are placed within the major division of flowering plants referred to as eudicots, specifically the core eudicots. This subgroup consists of the Dilleniaceae, superasterids and superrosids. The superrosids in turn have two components, rosids and Saxifragales.…

Ending:

> …indigenous Cotyledon pendens Van Jaarsv. endemic Cotyledon petiolaris Van Jaarsv. endemic Cotyledon rhombifolia Haw. accepted as Adromischus rhombifolius (Haw.) Lem. Cotyledon tomentosa Harv. indigenous Cotyledon tomentosa Harv. subsp. ladismithiensis (Poelln.) Toelken, endemic Cotyledon tomentosa Harv. subsp. tomentosa, endemic Cotyledon velutina Hook.f. endemic Cotyledon woodii Schonland & Baker f. endemic Cotyledon xanthantha Van Jaarsv. & Eggli, endemic Crassula Genus Crassula: Crassula acinaciformis Schinz, indigenous Crassula alba Forssk. var. alba, indigenous Crassula alba Forssk. var. pallida Toelken, indigenous Crassula alba Forssk. var. parvisepala (Schonland) Toelken, indigenous

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### First-wave feminism (`rec_4f36302ab33d12b202fbe231`)

Band/weight: B0 / 0.25; caps: ['general_linewise_list']; flags: ['linewise_list']

Chunk: 8/11; end boundary: line; characters: 7,914; paragraphs: 2; alpha: 0.778; repeated trigrams: 0.167

Beginning:

> First-wave feminism 1839 US, Mississippi: Mississippi was the first U.S. state that gave married women limited property rights. United Kingdom: The Custody of Infants Act 1839 made it possible for divorced mothers to be granted custody of their children under seven, but only if the Lord Chancellor agreed to it, and only if the mother was of good character. US, Mississippi: The Married Women's Property Act 1839 granted married women the right to own (but not control) property in their own name. 1840 US, Texas: Married women were allowed to own property in their own name. 1841 Bulgaria: The first secular girls school in Bulgaria was opened, making education and the profession of teacher avail…

Ending:

> …right to vote. New Zealand: New Zealand became the first self-governing country in the world in which all women had the right to vote in parliamentary elections. Cook Islands: The Cook Islands granted women the right to vote in island councils and a federal parliament. 1894 South Australia: South Australia granted women the right to vote. United Kingdom: The United Kingdom extended the right to vote in local elections to married women. 1895 US: Almost all U.S. states had passed some form of Sole Trader Laws, Property Laws, and Earnings Laws, granting married women the right to trade without their husbands' consent, own and/or control their own property, and control their own earnings. 1896

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### List of Saxifragales of South Africa (`rec_5cdcece6d4d8364c699018ba`)

Band/weight: B0 / 0.25; caps: ['general_linewise_list']; flags: ['linewise_list']

Chunk: 3/4; end boundary: line; characters: 7,969; paragraphs: 2; alpha: 0.812; repeated trigrams: 0.342

Beginning:

> List of Saxifragales of South Africa Crassula multiflora Schonland & Baker f. subsp. multiflora, endemic Crassula muricata Thunb. endemic Crassula muscosa L. indigenous Crassula muscosa L. var. muscosa, indigenous Crassula muscosa L. var. obtusifolia (Harv.) G.D.Rowley, indigenous Crassula muscosa L. var. parvula (Eckl. & Zeyh.) Toelken, endemic Crassula muscosa L. var. polpodacea (Eckl. & Zeyh.) G.D.Rowley, endemic Crassula namaquensis Schonland & Baker f. indigenous Crassula namaquensis Schonland & Baker f. subsp. comptonii (Hutch. & Pillans) Toelken, endemic Crassula namaquensis Schonland & Baker f. subsp. lutea (Schonland) Toelken, endemic Crassula namaquensis Schonland & Baker f. subsp…

Ending:

> …enous Crassula thunbergiana Schult. subsp. minutiflora (Schonland & Baker f.) Toelken, indigenous Crassula thunbergiana Schult. subsp. thunbergiana, endemic Crassula tomentosa Thunb. indigenous Crassula tomentosa Thunb. var. glabrifolia (Harv.) G.D.Rowley, indigenous Crassula tomentosa Thunb. var. tomentosa, indigenous Crassula tuberella Toelken, indigenous Crassula umbella Jacq. endemic Crassula umbellata Thunb. endemic Crassula umbraticola N.E.Br. indigenous Crassula vaginata Eckl. & Zeyh. indigenous Crassula vaginata Eckl. & Zeyh. subsp. vaginata, indigenous Crassula vaillantii (Willd.) Roth, not indigenous, naturalised Crassula vestita Thunb. endemic Crassula werneri N.Jacobsen, endemic

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### William Weintraub (`rec_7a6694c3fdf5952daf8c66e2`)

Band/weight: B0 / 0.25; caps: ['general_linewise_list']; flags: ['linewise_list']

Chunk: 1/2; end boundary: line; characters: 7,969; paragraphs: 15; alpha: 0.753; repeated trigrams: 0.125

Beginning:

> William Weintraub William Weintraub (February 19, 1926 – November 6, 2017) was a Canadian documentarian/filmmaker, journalist and author, best known for his long career with the National Film Board of Canada (NFB). Early life Weintraub was born in Montreal, to Louis Weintraub and Mina Blumer Weintraub, and grew up in the blue-collar neighbourhood of Verdun. His father had been a stock broker; he lost everything in the Wall Street Crash of 1929 and worked as the manager of a corner store. William studied English Literature and political science at McGill University, where he had worked on the McGill Daily. In 1947, he took the job of a ski reporter at The Montreal Gazette, from which he was…

Ending:

> …rector Background to Latin America - documentary short, James Beveridge 1963 - writer Canada Between Two World Wars - documentary short 1963 - writer and producer Canada: Human Vaccine - documentary short, Hector Lemieux 1963 - writer Canada: Beef Cattle - documentary short, Hector Lemieux 1963 - writer Canada: Calf Leather - documentary short, Hector Lemieux 1963 - writer Canada: Heating Units - documentary short, Hector Lemieux 1963 - writer The Visit - documentary short John Kemeny 1964 - writer Turn of the Century - documentary short 1964 - writer and producer Haida Carver - documentary short, Richard Gilbert 1964 - writer Landfall Asia - documentary short, Gordon Sparling 1964 - writer

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### List of Solanales of South Africa (`rec_ff59dda81ebf0224f65ec240`)

Band/weight: B0 / 0.25; caps: ['general_structured_low_prose', 'general_linewise_list']; flags: ['linewise_list', 'repetitive_language']

Chunk: 3/4; end boundary: line; characters: 7,996; paragraphs: 7; alpha: 0.832; repeated trigrams: 0.436

Beginning:

> List of Solanales of South Africa Lycium Genus Lycium: Lycium acutifolium E.Mey. ex Dunal, endemic Lycium afrum L. endemic Lycium amoenum Dammer, indigenous Lycium arenicola Miers, indigenous Lycium bosciifolium Schinz, indigenous Lycium cinereum Thunb. indigenous Lycium cordatum Mill. accepted as Carissa bispinosa (L.) Desf. ex Brenan, indigenous Lycium ferocissimum Miers, indigenous Lycium gariepense A.M.Venter, indigenous Lycium grandicalyx Joubert & Venter, indigenous Lycium hantamense A.M.Venter, indigenous Lycium hirsutum Dunal, indigenous Lycium horridum Thunb. indigenous Lycium mascarenense A.M.Venter & A.J.Scott, indigenous Lycium oxycarpum Dunal, endemic Lycium pilifolium C.H.Wrig…

Ending:

> …as Solanum laxum Spreng. not indigenous, naturalised Solanum kibweziense Dammer, accepted as Solanum tettense Klotzsch Solanum koniortodes Dammer, accepted as Solanum tettense Klotzsch Solanum kwebense N.E.Br. ex C.H.Wright, accepted as Solanum tettense Klotzsch, indigenous Solanum kwebense N.E.Br. ex C.H.Wright var. acutius Bitter, accepted as Solanum tettense Klotzsch Solanum kwebense N.E.Br. ex C.H.Wright var. chondropetalum (Dammer) Bitter, accepted as Solanum tettense Klotzsch Solanum kwebense N.E.Br. ex C.H.Wright var. luederitzii (Schinz) Bitter, accepted as Solanum tettense Klotzsch Solanum kwebense N.E.Br. ex C.H.Wright var. majorifrons Bitter, accepted as Solanum tettense Klotzsch

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:
