//tealium universal tag - utag.70 ut4.0.201406301940, Copyright 2014 Tealium.com Inc. All Rights Reserved.
if(typeof utag.ut=="undefined"){utag.ut={};}
utag.ut.libloader2=function(o,a,b,c,l){a=document;b=a.createElement('script');b.language='javascript';b.type='text/javascript';b.src=o.src;if(o.id){b.id=o.id};if(typeof o.cb=='function'){b.hFlag=0;b.onreadystatechange=function(){if((this.readyState=='complete'||this.readyState=='loaded')&&!b.hFlag){b.hFlag=1;o.cb()}};b.onload=function(){if(!b.hFlag){b.hFlag=1;o.cb()}}}
l=o.loc||'head';c=a.getElementsByTagName(l)[0];if(c){if(l=='script'){c.parentNode.insertBefore(b,c);}else{c.appendChild(b)}
utag.DB("Attach to "+l+": "+o.src)}}
try{(function(id,loader,u){u=utag.o[loader].sender[id]={};u.ev={'view':1};u.initialized=false;u.map={"TwitterPageID":"cus.pixel_id"};u.extend=[function(a,b,c,d){if(1){c=[b['js_page.USAA.ent.digitalData.page.attributes.uri'],b['js_page.USAA.ent.digitalData.page.pageID']];b['page_name']=c.join('?')}},function(a,b){if(typeof b['js_page.USAA.ent.digitalData.page.pageName']=='undefined'||b['js_page.USAA.ent.digitalData.page.pageName']==''){b['page_name']=b['page_name'].replace("/inet/","");}},function(a,b,c,d){c=['page_name'];for(d=0;d<c.length;d++){try{b[c[d]]=(b[c[d]]instanceof Array||b[c[d]]instanceof Object)?b[c[d]]:b[c[d]].toString().toLowerCase()}catch(e){}}},function(a,b,c,d,e,f,g){d=b['page_name'];if(typeof d=='undefined')return;c=[{'bank_cco/bkccoapplication?memberinformationpage':'l4flh'},{'bank_cco/bkccoapplication?bookedpage':'l4flj'},{'bank_cco/bkccoapplication?confirmationpage':'l4flj'},{'bank_cco/bkccoapplication?declinepage':'l4flk'},{'bank_cco/bkccoapplication?reviewpage':'l4flj'},{'bank_cco/bkccoapplication?declinewithcounteroffer':'l4flj'},{'ent_logon/logon?ent_login_member':'l4flh'},{'pages/ent_all_memorialday2014_landing_mkt?ent_all_memorialday2014_landing_mkt':'l4flk'},{'bk_cla/bkclambrapplication?applicationinformationpage':'l4flj'},{'bk_cla/bkclambrapplication?confirmationpage':'l4flk'},{'bk_cla/bkclambrapplication?offerspage':'l4flk'},{'bk_cla/bkclambrapplication?declinepage':'l4flk'},{'bk_cla/bkclambrapplication?reviewpage':'l4flk'},{'bk_cla/bkclambrapplication?systemcounterofferspage':'l4flk'},{'pages/car_buying_services_products?car_buying_services_products':'l4flh'},{'gas_pc_pas/gymemberautohistoryservlet?basicinfo':'l4flj'},{'gas_pc_pas/gymemberautohistoryservlet?slqtpolicyinfo':'l4flj'},{'gas_pc_pas/gymemberautohistoryservlet?slqtquoteresults':'l4fll'},{'gas_pc_pas/gymemberautohistoryservlet?dqissquoteresults':'l4fll'},{'gas_pc_pas/gymemberautohistoryservlet?slqtaddlinfo':'l4flk'},{'gas_pc_pas/gymemberautohistoryservlet?esignagreement':'l4flk'},{'gas_pc_pas/gymemberautohistoryservlet?slqtbinderconfirm':'l4flm'},{'gas_pc_pas/gymemberautohistoryservlet?binderconfirmation':'l4flm'}];var m=false;for(e=0;e<c.length;e++){for(f in c[e]){if(d.toString().indexOf(f)>-1){b['TwitterPageID']=c[e][f];m=true};};if(m)break};if(!m)b['TwitterPageID']='';}];u.send=function(a,b){if(u.ev[a]||typeof u.ev.all!="undefined"){var c,d,e,f;u.data={"pid":"","base_url":"//platform.twitter.com/oct.js"};for(c=0;c<u.extend.length;c++){try{d=u.extend[c](a,b);if(d==false)return}catch(e){}};for(d in utag.loader.GV(u.map)){if(typeof b[d]!="undefined"&&b[d]!=""){e=u.map[d].split(",");for(f=0;f<e.length;f++){u.data[e[f]]=b[d];}}}
u.twitter_callback=function(){u.initialized=true;twttr.conversion.trackPid(b['TwitterPageID']);};if(!u.initialized){utag.ut.libloader2({src:u.data.base_url,cb:u.twitter_callback});}else{u.twitter_callback();}
}}
utag.o[loader].loader.LOAD(id);})('70','usaa.main');}catch(e){}