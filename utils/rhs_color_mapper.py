import argparse
import numpy as np
from skimage import color
# Color informatino extracted from https://azaleas.org/rhs-color-fan-1/
# Archived https://web.archive.org/web/20230923032935/https://azaleas.org/rhs-color-fan-1/


# Define RGB color mapping
RGB = {
  '1A': 'e2db23',
  '1B': 'e2db2f',
  '1C': 'e4dc6c',
  '1D': 'f0e18d',
  '2A': 'f3dc19',
  '2B': 'ecdd32',
  '2C': 'edea73',
  '2D': 'f0e098',
  '3A': 'f8da21',
  '3B': 'f8e533',
  '3C': 'f0e55f',
  '3D': 'efe981',
  '4A': 'fbe432',
  '4B': 'f8e95a',
  '4C': 'f9e67e',
  '4D': 'f2e9bb',
  '5A': 'eecc20',
  '5B': 'fbd923',
  '5C': 'ffe151',
  '5D': 'ffe383',
  '6A': 'f9cf21',
  '6B': 'fed225',
  '6C': 'ffd73d',
  '6D': 'fddf75',
  '7A': 'eab814',
  '7B': 'edc12d',
  '7C': 'ebc72e',
  '7D': 'e6c749',
  '8A': 'ffd534',
  '8B': 'ffdb51',
  '8C': 'f8e775',
  '8D': 'fce7b0',
  '9A': 'ffc912',
  '9B': 'ffcb19',
  '9C': 'ffd945',
  '9D': 'fce393',
  '10A': 'ffd045',
  '10B': 'ffd557',
  '10C': 'fdda70',
  '10D': 'f7e49e',
  '11A': 'fec63a',
  '11B': 'fad065',
  '11C': 'fad890',
  '11D': 'ffe4c3',
  '12A': 'ffc813',
  '12B': 'ffce2d',
  '12C': 'ffd85c',
  '12D': 'fddd8f',
  '13A': 'ffbe0d',
  '13B': 'ffc320',
  '13C': 'ffca39',
  '13D': 'f7de99',
  '14A': 'ffb100',
  '14B': 'ffb700',
  '14C': 'ffcb47',
  '14D': 'ffd076',
  '15A': 'ffb010',
  '15B': 'ffb61f',
  '15C': 'ffbc2e',
  '15D': 'ffcd5f',
  '16A': 'ffb021',
  '16B': 'ffb047',
  '16C': 'ffb759',
  '16D': 'ffc480',
  '17A': 'ff9000',
  '17B': 'ffa000',
  '17C': 'ffa908',
  '17D': 'ffbf4f',
  '18A': 'ffbe4e',
  '18B': 'ffcd81',
  '18C': 'ffd4a5',
  '18D': 'ffdaa2',
  '19A': 'ffb23c',
  '19B': 'ffb27e',
  '19C': 'ffce91',
  '19D': 'ffd8bf',
  '20A': 'ff9f2a',
  '20B': 'ffbb52',
  '20C': 'ffc681',
  '20D': 'fdd5aa',
  '21A': 'ff9700',
  '21B': 'ff9f10',
  '21C': 'ffae2a',
  '21D': 'ffbd58',
  '22A': 'fb8721',
  '22B': 'ffa644',
  '22C': 'ffab54',
  '22D': 'ffb469',
  '23A': 'ff8700',
  '23B': 'ff9b1c',
  '23C': 'ffac4b',
  '23D': 'ffc790',
  '24A': 'ff7c11',
  '24B': 'ff8929',
  '24C': 'ffa558',
  '24D': 'ffb680',
  '25A': 'ff6a11',
  '25B': 'ff7c1d',
  '25C': 'ff8c36',
  '25D': 'ffad6e',
  '26A': 'ef681c',
  '26B': 'fc7b31',
  '26C': 'fd9259',
  '26D': 'fca074',
  '27A': 'ffc09c',
  '27B': 'ffc9b5',
  '27C': 'ffd1c6',
  '27D': 'ffdad2',
  '28A': 'ff5c0f',
  '28B': 'ff7117',
  '28C': 'ff934d',
  '28D': 'ffa26a',
  '29A': 'ff873f',
  '29B': 'ff925d',
  '29C': 'ffa682',
  '29D': 'ecb8a3',
  '30A': 'f84423',
  '30B': 'ff4a20',
  '30C': 'ff5119',
  '30D': 'ff6c2f',
  '31A': 'e2401d',
  '31B': 'ed532b',
  '31C': 'e76c51',
  '31D': 'f9826b',
  '32A': 'ef3610',
  '32B': 'f64f2c',
  '32C': 'ff623b',
  '32D': 'ff8665',
  '33A': 'e02913',
  '33B': 'f13918',
  '33C': 'ff5d3c',
  '33D': 'ff8871',
  '34A': 'f82f20',
  '34B': 'ea3c23',
  '34C': 'fd5130',
  '34D': 'e8674b',
  '35A': 'c53a25',
  '35B': 'dd4637',
  '35C': 'df645b',
  '35D': 'f08285',
  '36A': 'fca29c',
  '36B': 'f7adb0',
  '36C': 'ffb5bb',
  '36D': 'fdd0d4',
  '37A': 'e75e48',
  '37B': 'f27a6c',
  '37C': 'f99488',
  '37D': 'ffaca1',
  '38A': 'ff706b',
  '38B': 'f98a84',
  '38C': 'fc9d93',
  '38D': 'ffb6b0',
  '39A': 'd2332e',
  '39B': 'e34b48',
  '39C': 'f58684',
  '39D': 'ffa4ab',
  '40A': 'ec2b23',
  '40B': 'e52f1e',
  '40C': 'e7422b',
  '40D': 'f8573a',
  '41A': 'c11c16',
  '41B': 'c32826',
  '41C': 'b13e36',
  '41D': 'bb4e4c',
  '42A': 'b91715',
  '42B': 'bd2c1d',
  '42C': 'cb3324',
  '42D': 'd14637',
  '43A': 'c61914',
  '43B': 'de241f',
  '43C': 'e83639',
  '43D': 'ff646a',
  '44A': 'af140d',
  '44B': 'c01811',
  '44C': 'd42923',
  '44D': 'e63c3d',
  '45A': 'a20712',
  '45B': 'b30313',
  '45C': 'b41920',
  '45D': 'c8252d',
  '46A': '7f0e18',
  '46B': '971013',
  '46C': 'b91620',
  '46D': 'dc2e30',
  '47A': '9e2122',
  '47B': 'b9242a',
  '47C': 'da3742',
  '47D': 'df4253',
  '48A': 'c84042',
  '48B': 'da4f4f',
  '48C': 'f26069',
  '48D': 'ff8591',
  '49A': 'f87583',
  '49B': 'f68e97',
  '49C': 'fab0be',
  '49D': 'f0c9d6',
  '50A': 'b81f2c',
  '50B': 'd43041',
  '50C': 'e45b6a',
  '50D': 'f59eab',
  '51A': 'ba263e',
  '51B': 'd23e53',
  '51C': 'd85c6b',
  '51D': 'dd7f8a',
  '52A': 'bf1f3a',
  '52B': 'd83151',
  '52C': 'db4266',
  '52D': 'ee6982',
  '53A': '720d1b',
  '53B': '8d1523',
  '53C': 'a81a30',
  '53D': 'bb203d',
  '54A': 'bb3151',
  '54B': 'c53a5d',
  '54C': 'd5607a',
  '54D': 'df86a1',
  '55A': 'd03768',
  '55B': 'df5a8d',
  '55C': 'f184a9',
  '55D': 'f5a1c1',
  '56A': 'df99b4',
  '56B': 'e1a2be',
  '56C': 'd8abc2',
  '56D': 'd9b5ca',
  '57A': 'a10328',
  '57B': 'a5002c',
  '57C': 'b40e44',
  '57D': 'c1285f',
  '58A': '741133',
  '58B': 'b11f50',
  '58C': 'c72e62',
  '58D': 'da5381',
  '59A': '4b0f1d',
  '59B': '600c23',
  '59C': '7b1535',
  '59D': 'a42650',
  '60A': '6d0b22',
  '60B': '790a2b',
  '60C': '89143a',
  '60D': 'a4224f',
  '61A': '670a2d',
  '61B': '8d113e',
  '61C': 'b71f53',
  '61D': 'dc3e7c',
  '62A': 'e55fa0',
  '62B': 'e677ad',
  '62C': 'eb8ebc',
  '62D': 'eaaed4',
  '63A': '9f1843',
  '63B': 'b8285f',
  '63C': 'dc5b8f',
  '63D': 'e981ab',
  '64A': '781239',
  '64B': '931a49',
  '64C': 'b5316e',
  '64D': 'c5568b',
  '65A': 'db76b0',
  '65B': 'e496c4',
  '65C': 'eea7ce',
  '65D': 'e3b5df',
  '66A': 'b80049',
  '66B': 'bc004e',
  '66C': 'c63b73',
  '66D': 'd05f91',
  '67A': '9f1b4c',
  '67B': 'bc1e6a',
  '67C': 'cd2c7b',
  '67D': 'd35ca3',
  '68A': 'c5388b',
  '68B': 'db57ae',
  '68C': 'e47bc4',
  '68D': 'de96d4',
  '69A': 'eab7eb',
  '69B': 'e7bdee',
  '69C': 'e8c8f1',
  '69D': 'e7c9f2',
  '70A': '761c4c',
  '70B': '9e2970',
  '70C': 'b24f89',
  '70D': 'cf7fb9',
  '71A': '5e0e35',
  '71B': '85184d',
  '71C': 'a01d5f',
  '71D': 'b62e7d',
  '72A': '6d124a',
  '72B': '7f1d60',
  '72C': 'ac3a8f',
  '72D': 'bc4da0',
  '73A': 'bd3f97',
  '73B': 'd667b9',
  '73C': 'eb8bd8',
  '73D': 'd5b3e5',
  '74A': '950063',
  '74B': '9f1a70',
  '74C': 'a95b96',
  '74D': 'c37eb3',
  '75A': 'a35eb2',
  '75B': 'bc7ecd',
  '75C': 'c894cd',
  '75D': 'cfb0e0',
  '76A': 'b373c0',
  '76B': 'd28fd6',
  '76C': 'e1a9e4',
  '76D': 'e6c4ec',
  '77A': '531752',
  '77B': '85398d',
  '77C': 'a367b6',
  '77D': 'b581ca',
  '78A': '7e1575',
  '78B': '81287c',
  '78C': '975898',
  '78D': 'b779b1',
  '79A': '2a172c',
  '79B': '36163b',
  '79C': '4a1c4e',
  '79D': '5b2e5f',
  '80A': '660f6c',
  '80B': '711f7b',
  '80C': '94509b',
  '80D': 'a97db4',
  '81A': '5f1179',
  '81B': '6e228a',
  '81C': '8f55a4',
  '81D': 'a878bc',
  '82A': '4f0f75',
  '82B': '551a79',
  '82C': '7b4c9a',
  '82D': '9f6eb3',
  '83A': '2e1843',
  '83B': '3d1a55',
  '83C': '482366',
  '83D': '602e80',
  '84A': '6f409a',
  '84B': '9666be',
  '84C': 'b685ca',
  '84D': 'd6a9db',
  '85A': '815dae',
  '85B': '9877c6',
  '85C': 'b393dd',
  '85D': 'd4b8e8',
  '86A': '2d174b',
  '86B': '452575',
  '86C': '57308a',
  '86D': '6a44a6',
  '87A': '4a1282',
  '87B': '58248b',
  '87C': '8b62b2',
  '87D': 'aa84cb',
  '88A': '451389',
  '88B': '4c1a8a',
  '88C': '7052a8',
  '88D': '917abe',
  '89A': '20005a',
  '89B': '260463',
  '89C': '331574',
  '89D': '543f90',
  '90A': '341e66',
  '90B': '41297b',
  '90C': '543791',
  '90D': '6044a4',
  '91A': '7455ac',
  '91B': '9d7ecd',
  '91C': 'bb9bdb',
  '91D': 'cdbaeb',
  '92A': '513e9d',
  '92B': '7f6dc4',
  '92C': 'a692de',
  '92D': 'c4b8e1',
  '93A': '1c1051',
  '93B': '2f2074',
  '93C': '422f8f',
  '93D': '6b5ebc',
  '94A': '2b1f82',
  '94B': '3e39a1',
  '94C': '5a57bd',
  '94D': '7672d6',
  '95A': '191360',
  '95B': '002381',
  '95C': '2d3897',
  '95D': '8d8dcf',
  '96A': '181978',
  '96B': '2c2a9b',
  '96C': '3135a7',
  '96D': '4f52bd',
  '97A': '5260c4',
  '97B': '6d7fce',
  '97C': '97a1e0',
  '97D': 'c6bceb',
  '98A': '233183',
  '98B': '374490',
  '98C': '4c60ae',
  '98D': '6879ba',
  '99A': '121353',
  '99B': '151574',
  '99C': '101f9d',
  '99D': '371eb7',
  '100A': '1723a0',
  '100B': '4255be',
  '100C': '6673d1',
  '100D': '9d9ee6',
  '101A': '0438ba',
  '101B': '3350c8',
  '101C': '5c79d5',
  '101D': '9ba9e4',
  '102A': '101852',
  '102B': '13296c',
  '102C': '1e2d7b',
  '102D': '314290',
  '103A': '191830',
  '103B': '171b3e',
  '103C': '151d46',
  '103D': '192d5f',
  '104A': '0c298a',
  '104B': '2248b0',
  '104C': '325cc8',
  '104D': '5d85dd',
  '105A': '011e66',
  '105B': '002984',
  '105C': '00399c',
  '105D': '1e55c4',
  '106A': '326ddc',
  '106B': '5686e8',
  '106C': '80a1e7',
  '106D': 'a5b5f1',
  '107A': '003e9e',
  '107B': '1b5ac1',
  '107C': '5384d8',
  '107D': '6b90d9',
  '108A': 'a0b8e7',
  '108B': 'b0c7f7',
  '108C': 'b5c6f4',
  '108D': 'c6ccf5',
  '109A': '0043c5',
  '109B': '0050c8',
  '109C': '256dd4',
  '109D': '4c8ae0',
  '110A': '004ca9',
  '110B': '0063bd',
  '110C': '58a1ed',
  '110D': '74b2f2',
  '111A': '004e96',
  '111B': '1781c8',
  '111C': '4b98d9',
  '111D': '75aae0',
  '112A': '9ab5dd',
  '112B': 'abbfe6',
  '112C': 'b7c7e7',
  '112D': 'cccce8',
  '113A': '005593',
  '113B': '106aae',
  '113C': '5691d8',
  '113D': 'a3c2e0',
  '114A': '002547',
  '114B': '00355e',
  '114C': '004976',
  '114D': '115e90',
  '115A': '275483',
  '115B': '306697',
  '115C': '587ca5',
  '115D': '849fc6',
  '116A': '123259',
  '116B': '183d66',
  '116C': '244f78',
  '116D': '3f6488',
  '117A': '5590c1',
  '117B': '72a8cd',
  '117C': '8cb9d9',
  '117D': 'a3bed8',
  '118A': '006095',
  '118B': '1478af',
  '118C': '4a96c6',
  '118D': 'a3c7e5',
  '119A': '29597b',
  '119B': '3b6c92',
  '119C': '5181a9',
  '119D': '719fc5',
  '120A': '006285',
  '120B': '0d7195',
  '120C': '4890af',
  '120D': '94b8ce',
  '121A': '1f6e91',
  '121B': '428db1',
  '121C': '6ba1c1',
  '121D': '9ab6d5',
  '122A': '345f6f',
  '122B': '496e83',
  '122C': '7394a6',
  '122D': 'a0bacd',
  '123A': '52a0ae',
  '123B': '68b1c1',
  '123C': '96c2d0',
  '123D': 'afccd0',
  '124A': '00696c',
  '124B': '21868c',
  '124C': '589ba1',
  '124D': 'a6c3cc',
  '125A': '005f50',
  '125B': '007a6c',
  '125C': '38b3ad',
  '125D': '65c5c1',
  '126A': '003638',
  '126B': '004b4b',
  '126C': '0c5a59',
  '126D': '2b7776',
  '127A': '0d413d',
  '127B': '00574d',
  '127C': '006e60',
  '127D': '278375',
  '128A': '007554',
  '128B': '298f79',
  '128C': '64ac9d',
  '128D': 'aecdc8',
  '129A': '007b51',
  '129B': '34997b',
  '129C': '4cb292',
  '129D': '91c6b4',
  '130A': '088b4d',
  '130B': '48a77a',
  '130C': '70b897',
  '130D': 'b4d1c3',
  '131A': '102724',
  '131B': '0e3326',
  '131C': '02412d',
  '131D': '0b6344',
  '132A': '002f1f',
  '132B': '003d26',
  '132C': '095a40',
  '132D': '34875d',
  '133A': '142a26',
  '133B': '204c45',
  '133C': '496a66',
  '133D': '5f8f84',
  '134A': '00621e',
  '134B': '229143',
  '134C': '4fac69',
  '134D': '7fbf90',
  '135A': '06291b',
  '135B': '0e3822',
  '135C': '226834',
  '135D': '659e63',
  '136A': '10251c',
  '136B': '183422',
  '136C': '3b6545',
  '136D': '87ae8c',
  '137A': '25361d',
  '137B': '2d3f23',
  '137C': '374c27',
  '137D': '3e5b30',
  '138A': '344d29',
  '138B': '52673b',
  '138C': '768c62',
  '138D': '8fa476',
  '139A': '14291a',
  '139B': '2d4e2c',
  '139C': '526f3d',
  '139D': '869f63',
  '140A': '067d10',
  '140B': '3a9a2e',
  '140C': '64b65c',
  '140D': '95c98d',
  '141A': '163d11',
  '141B': '114412',
  '141C': '3a7229',
  '141D': '71a247',
  '142A': '6cae32',
  '142B': '7bb14f',
  '142C': 'a1c77b',
  '142D': 'bad89f',
  '143A': '31591a',
  '143B': '41671a',
  '143C': '547424',
  '143D': '7fa55e',
  '144A': '475b0d',
  '144B': '6a831f',
  '144C': '7d9728',
  '144D': 'a3bf5d',
  '145A': '80962d',
  '145B': '95ab46',
  '145C': 'b9c171',
  '145D': 'c8cf9b',
  '146A': '343b16',
  '146B': '474f21',
  '146C': '5b6327',
  '146D': '727333',
  '147A': '232c1b',
  '147B': '434928',
  '147C': '6c7345',
  '147D': '949a6c',
  '148A': '404723',
  '148B': '555b37',
  '148C': '7c7a56',
  '148D': '918d6a',
  '149A': '90c222',
  '149B': '9bc842',
  '149C': 'add664',
  '149D': 'd5e1ac',
  '150A': 'b5cd1f',
  '150B': 'c3d238',
  '150C': 'd0dd5a',
  '150D': 'e5e3a1',
  '151A': '999019',
  '151B': 'b0a610',
  '151C': 'bab719',
  '151D': 'c1be27',
  '152A': '554816',
  '152B': '5e5413',
  '152C': '7e6510',
  '152D': '8c751f',
  '153A': '8e790d',
  '153B': 'a58410',
  '153C': 'ab8b13',
  '153D': 'c39a27',
  '154A': 'c4ce14',
  '154B': 'd3d429',
  '154C': 'd5dc5a',
  '154D': 'e0e17b',
  '155A': 'e9ddcf',
  '155B': 'f6eae3',
  '155C': 'e8e2df',
  '155D': 'f2e5e4',
  '156A': 'b1a28d',
  '156B': 'bfb09e',
  '156C': 'cab7a8',
  '156D': 'd3c3ba',
  '157A': 'd0cbad',
  '157B': 'd8d3bb',
  '157C': 'e2dac7',
  '157D': 'eae2dc',
  '158A': 'f2cd9e',
  '158B': 'fde0b9',
  '158C': 'fbe5cb',
  '158D': 'f9e4d6',
  '159A': 'f7b599',
  '159B': 'fbc3a9',
  '159C': 'f5cab9',
  '159D': 'fbd8d2',
  '160A': 'cca947',
  '160B': 'd6b45f',
  '160C': 'd6bd7a',
  '160D': 'd2b887',
  '161A': 'c4924c',
  '161B': 'df9d4f',
  '161C': 'e0aa72',
  '161D': 'debe8c',
  '162A': 'd19438',
  '162B': 'e0a84a',
  '162C': 'daad61',
  '162D': 'ddb57c',
  '163A': 'b95d0b',
  '163B': 'd67212',
  '163C': 'dd953b',
  '163D': 'e7b36b',
  '164A': '90451c',
  '164B': 'b96d30',
  '164C': 'd08a41',
  '164D': 'e6b781',
  '165A': '4b281d',
  '165B': '92481f',
  '165C': 'bf733f',
  '165D': 'e19f6b',
  '166A': '3a2521',
  '166B': '6f2d1d',
  '166C': '914423',
  '166D': 'a65d34',
  '167A': 'ad4f16',
  '167B': 'c05d23',
  '167C': 'ce6a2b',
  '167D': 'd37638',
  '168A': 'c0421c',
  '168B': 'cc4b1a',
  '168C': 'd96626',
  '168D': 'e3884b',
  '169A': 'b52d15',
  '169B': 'c03e15',
  '169C': 'ce460e',
  '169D': 'de5216',
  '170A': 'c0471e',
  '170B': 'c75625',
  '170C': 'd26c38',
  '170D': 'e08854',
  '171A': '93341b',
  '171B': 'ab3f1a',
  '171C': 'b44f2a',
  '171D': 'cb6e46',
  '172A': '6d2716',
  '172B': '7d2e1c',
  '172C': '9f3f22',
  '172D': 'b35428',
  '173A': '792822',
  '173B': 'a73b26',
  '173C': 'b55739',
  '173D': 'd37d5a',
  '174A': '6a2e24',
  '174B': '93432e',
  '174C': 'a8533c',
  '174D': 'bd654b',
  '175A': '5a211b',
  '175B': '6a2117',
  '175C': '802918',
  '175D': '903829',
  '176A': '602221',
  '176B': '6b2d2a',
  '176C': '7c352d',
  '176D': '9b4334',
  '177A': '532723',
  '177B': '6d3a2f',
  '177C': '864336',
  '177D': '9b5346',
  '178A': '601b1b',
  '178B': '6c2019',
  '178C': '8c2b1f',
  '178D': 'a63728',
  '179A': '961d22',
  '179B': 'b43431',
  '179C': 'd25c4f',
  '179D': 'f68c7a',
  '180A': '862524',
  '180B': 'a02c2c',
  '180C': 'ae3a39',
  '180D': 'c55352',
  '181A': '762124',
  '181B': '892b2d',
  '181C': '9f3c3e',
  '181D': 'b76061',
  '182A': '7c252c',
  '182B': '873a3c',
  '182C': 'a35255',
  '182D': 'ae5a5f',
  '183A': '51161a',
  '183B': '57191d',
  '183C': '671b23',
  '183D': '742530',
  '184A': '641c21',
  '184B': '6e212c',
  '184C': 'a32d42',
  '184D': 'b73850',
  '185A': '6e1118',
  '185B': '7d1e27',
  '185C': '8b2c41',
  '185D': 'a8425b',
  '186A': '7f223c',
  '186B': '963152',
  '186C': 'ab5479',
  '186D': 'c87a9c',
  '187A': '2a0f15',
  '187B': '460d17',
  '187C': '560d1c',
  '187D': '670e2c',
  '188A': '71797e',
  '188B': '7d8689',
  '188C': '929a9f',
  '188D': 'adaab1',
  '189A': '334633',
  '189B': '5a756b',
  '189C': '849b91',
  '189D': 'a4b7b2',
  '190A': '7b8d7e',
  '190B': '939b94',
  '190C': 'a5a8a7',
  '190D': 'c1c2c5',
  '191A': '505c45',
  '191B': '627153',
  '191C': '938c97',
  '191D': 'a5a49a',
  '192A': 'a9b29e',
  '192B': 'aebaa7',
  '192C': 'bfc9bb',
  '192D': 'ccd4cd',
  '193A': 'a0a381',
  '193B': 'aab394',
  '193C': 'bbc0ae',
  '193D': 'c2c3b4',
  '194A': '75775a',
  '194B': '848461',
  '194C': '9f9892',
  '194D': 'b1ab97',
  '195A': '797253',
  '195B': '978f70',
  '195C': 'b1a78e',
  '195D': 'bbb29a',
  '196A': '9e9988',
  '196B': 'b6a999',
  '196C': 'bcb6a8',
  '196D': 'c9c1bb',
  '197A': '544938',
  '197B': '625645',
  '197C': '78725b',
  '197D': '959381',
  '198A': '6f6c60',
  '198B': '807e76',
  '198C': '8c8781',
  '198D': 'a69e9f',
  '199A': '583c1e',
  '199B': '725332',
  '199C': '855f41',
  '199D': '937255',
  '200A': '231818',
  '200B': '2d1b19',
  '200C': '351f1a',
  '200D': '47291f',
  '201A': '574746',
  '201B': '6d5c5c',
  '201C': '7e6970',
  '201D': '917c84',
  '202A': '181417',
  '202B': '48474f',
  '202C': '74787f',
  '202D': 'b4bbc3'
}

# Define UCL values mapping
UCL = {
  '1A': '98',
  '1B': '98',
  '1C': '101',
  '1D': '104',
  '2A': '97',
  '2B': '98',
  '2C': '119',
  '2D': '104',
  '3A': '98',
  '3B': '98',
  '3C': '101',
  '3D': '101',
  '4A': '98',
  '4B': '101',
  '4C': '101',
  '4D': '121',
  '5A': '98',
  '5B': '98',
  '5C': '101',
  '5D': '101',
  '6A': '98',
  '6B': '98',
  '6C': '98',
  '6D': '101',
  '7A': '83',
  '7B': '83',
  '7C': '98',
  '7D': '101',
  '8A': '83',
  '8B': '101',
  '8C': '101',
  '8D': '89',
  '9A': '82',
  '9B': '82',
  '9C': '83',
  '9D': '104',
  '10A': '83',
  '10B': '86',
  '10C': '86',
  '10D': '104',
  '11A': '83',
  '11B': '86',
  '11C': '89',
  '11D': '89',
  '12A': '82',
  '12B': '83',
  '12C': '86',
  '12D': '89',
  '13A': '82',
  '13B': '83',
  '13C': '83',
  '13D': '104',
  '14A': '82',
  '14B': '82',
  '14C': '83',
  '14D': '86',
  '15A': '82',
  '15B': '82',
  '15C': '83',
  '15D': '86',
  '16A': '82',
  '16B': '70',
  '16C': '70',
  '16D': '73',
  '17A': '68',
  '17B': '82',
  '17C': '82',
  '17D': '86',
  '18A': '86',
  '18B': '86',
  '18C': '89',
  '18D': '89',
  '19A': '70',
  '19B': '28',
  '19C': '89',
  '19D': '73',
  '20A': '83',
  '20B': '86',
  '20C': '89',
  '20D': '89',
  '21A': '66',
  '21B': '67',
  '21C': '83',
  '21D': '86',
  '22A': '68',
  '22B': '70',
  '22C': '70',
  '22D': '70',
  '23A': '66',
  '23B': '67',
  '23C': '70',
  '23D': '73',
  '24A': '50',
  '24B': '68',
  '24C': '70',
  '24D': '73',
  '25A': '50',
  '25B': '50',
  '25C': '49',
  '25D': '70',
  '26A': '50',
  '26B': '50',
  '26C': '52',
  '26D': '28',
  '27A': '28',
  '27B': '28',
  '27C': '31',
  '27D': '31',
  '28A': '25',
  '28B': '48',
  '28C': '52',
  '28D': '52',
  '29A': '49',
  '29B': '52',
  '29C': '28',
  '29D': '31',
  '30A': '34',
  '30B': '34',
  '30C': '25',
  '30D': '50',
  '31A': '35',
  '31B': '35',
  '31C': '26',
  '31D': '29',
  '32A': '34',
  '32B': '35',
  '32C': '26',
  '32D': '26',
  '33A': '34',
  '33B': '34',
  '33C': '26',
  '33D': '29',
  '34A': '34',
  '34B': '34',
  '34C': '35',
  '34D': '26',
  '35A': '37',
  '35B': '37',
  '35C': '26',
  '35D': '5',
  '36A': '28',
  '36B': '28',
  '36C': '28',
  '36D': '31',
  '37A': '26',
  '37B': '26',
  '37C': '29',
  '37D': '28',
  '38A': '26',
  '38B': '29',
  '38C': '28',
  '38D': '28',
  '39A': '12',
  '39B': '27',
  '39C': '29',
  '39D': '4',
  '40A': '34',
  '40B': '34',
  '40C': '35',
  '40D': '26',
  '41A': '34',
  '41B': '12',
  '41C': '37',
  '41D': '27',
  '42A': '34',
  '42B': '35',
  '42C': '35',
  '42D': '37',
  '43A': '34',
  '43B': '34',
  '43C': '27',
  '43D': '26',
  '44A': '11',
  '44B': '34',
  '44C': '34',
  '44D': '27',
  '45A': '11',
  '45B': '11',
  '45C': '11',
  '45D': '12',
  '46A': '12',
  '46B': '11',
  '46C': '11',
  '46D': '27',
  '47A': '15',
  '47B': '12',
  '47C': '27',
  '47D': '3',
  '48A': '3',
  '48B': '3',
  '48C': '2',
  '48D': '2',
  '49A': '2',
  '49B': '5',
  '49C': '4',
  '49D': '7',
  '50A': '12',
  '50B': '3',
  '50C': '2',
  '50D': '4',
  '51A': '12',
  '51B': '3',
  '51C': '3',
  '51D': '5',
  '52A': '11',
  '52B': '3',
  '52C': '3',
  '52D': '2',
  '53A': '13',
  '53B': '12',
  '53C': '12',
  '53D': '12',
  '54A': '255',
  '54B': '248',
  '54C': '2',
  '54D': '250',
  '55A': '248',
  '55B': '247',
  '55C': '249',
  '55D': '252',
  '56A': '252',
  '56B': '252',
  '56C': '252',
  '56D': '252',
  '57A': '11',
  '57B': '254',
  '57C': '254',
  '57D': '254',
  '58A': '258',
  '58B': '255',
  '58C': '255',
  '58D': '248',
  '59A': '16',
  '59B': '256',
  '59C': '258',
  '59D': '255',
  '60A': '13',
  '60B': '255',
  '60C': '255',
  '60D': '255',
  '61A': '256',
  '61B': '255',
  '61C': '254',
  '61D': '248',
  '62A': '247',
  '62B': '250',
  '62C': '249',
  '62D': '252',
  '63A': '255',
  '63B': '255',
  '63C': '247',
  '63D': '249',
  '64A': '258',
  '64B': '255',
  '64C': '255',
  '64D': '248',
  '65A': '250',
  '65B': '249',
  '65C': '252',
  '65D': '252',
  '66A': '254',
  '66B': '254',
  '66C': '248',
  '66D': '250',
  '67A': '255',
  '67B': '254',
  '67C': '248',
  '67D': '247',
  '68A': '248',
  '68B': '247',
  '68C': '250',
  '68D': '249',
  '69A': '226',
  '69B': '226',
  '69C': '226',
  '69D': '226',
  '70A': '258',
  '70B': '237',
  '70C': '248',
  '70D': '250',
  '71A': '256',
  '71B': '255',
  '71C': '255',
  '71D': '255',
  '72A': '255',
  '72B': '237',
  '72C': '237',
  '72D': '248',
  '73A': '248',
  '73B': '247',
  '73C': '249',
  '73D': '226',
  '74A': '236',
  '74B': '236',
  '74C': '240',
  '74D': '250',
  '75A': '222',
  '75B': '222',
  '75C': '221',
  '75D': '226',
  '76A': '222',
  '76B': '221',
  '76C': '226',
  '76D': '226',
  '77A': '238',
  '77B': '218',
  '77C': '222',
  '77D': '222',
  '78A': '236',
  '78B': '237',
  '78C': '222',
  '78D': '250',
  '79A': '224',
  '79B': '224',
  '79C': '219',
  '79D': '223',
  '80A': '216',
  '80B': '218',
  '80C': '222',
  '80D': '222',
  '81A': '216',
  '81B': '216',
  '81C': '222',
  '81D': '222',
  '82A': '216',
  '82B': '218',
  '82C': '223',
  '82D': '222',
  '83A': '224',
  '83B': '219',
  '83C': '218',
  '83D': '218',
  '84A': '218',
  '84B': '222',
  '84C': '222',
  '84D': '226',
  '85A': '222',
  '85B': '222',
  '85C': '221',
  '85D': '226',
  '86A': '211',
  '86B': '207',
  '86C': '207',
  '86D': '206',
  '87A': '205',
  '87B': '216',
  '87C': '222',
  '87D': '222',
  '88A': '205',
  '88B': '205',
  '88C': '210',
  '88D': '210',
  '89A': '205',
  '89B': '205',
  '89C': '205',
  '89D': '210',
  '90A': '207',
  '90B': '207',
  '90C': '206',
  '90D': '206',
  '91A': '210',
  '91B': '210',
  '91C': '226',
  '91D': '213',
  '92A': '206',
  '92B': '210',
  '92C': '210',
  '92D': '226',
  '93A': '208',
  '93B': '207',
  '93C': '207',
  '93D': '210',
  '94A': '207',
  '94B': '196',
  '94C': '195',
  '94D': '210',
  '95A': '194',
  '95B': '176',
  '95C': '196',
  '95D': '199',
  '96A': '194',
  '96B': '194',
  '96C': '196',
  '96D': '195',
  '97A': '195',
  '97B': '199',
  '97C': '202',
  '97D': '226',
  '98A': '196',
  '98B': '182',
  '98C': '181',
  '98D': '199',
  '99A': '197',
  '99B': '194',
  '99C': '194',
  '99D': '205',
  '100A': '194',
  '100B': '178',
  '100C': '199',
  '100D': '198',
  '101A': '176',
  '101B': '178',
  '101C': '181',
  '101D': '202',
  '102A': '179',
  '102B': '182',
  '102C': '178',
  '102D': '182',
  '103A': '204',
  '103B': '183',
  '103C': '183',
  '103D': '182',
  '104A': '178',
  '104B': '178',
  '104C': '177',
  '104D': '181',
  '105A': '179',
  '105B': '178',
  '105C': '178',
  '105D': '178',
  '106A': '177',
  '106B': '181',
  '106C': '180',
  '106D': '202',
  '107A': '178',
  '107B': '178',
  '107C': '181',
  '107D': '181',
  '108A': '181',
  '108B': '181',
  '108C': '181',
  '108D': '181',
  '109A': '176',
  '109B': '176',
  '109C': '177',
  '109D': '181',
  '110A': '178',
  '110B': '178',
  '110C': '181',
  '110D': '180',
  '111A': '178',
  '111B': '168',
  '111C': '172',
  '111D': '184',
  '112A': '184',
  '112B': '184',
  '112C': '184',
  '112D': '189',
  '113A': '169',
  '113B': '168',
  '113C': '181',
  '113D': '184',
  '114A': '174',
  '114B': '173',
  '114C': '169',
  '114D': '169',
  '115A': '182',
  '115B': '181',
  '115C': '185',
  '115D': '185',
  '116A': '182',
  '116B': '182',
  '116C': '182',
  '116D': '172',
  '117A': '172',
  '117B': '171',
  '117C': '171',
  '117D': '184',
  '118A': '169',
  '118B': '168',
  '118C': '172',
  '118D': '184',
  '119A': '173',
  '119B': '172',
  '119C': '172',
  '119D': '172',
  '120A': '169',
  '120B': '168',
  '120C': '172',
  '120D': '184',
  '121A': '172',
  '121B': '172',
  '121C': '172',
  '121D': '184',
  '122A': '173',
  '122B': '172',
  '122C': '218',
  '122D': '184',
  '123A': '163',
  '123B': '162',
  '123C': '162',
  '123D': '148',
  '124A': '160',
  '124B': '163',
  '124C': '163',
  '124D': '148',
  '125A': '160',
  '125B': '159',
  '125C': '159',
  '125D': '162',
  '126A': '164',
  '126B': '164',
  '126C': '164',
  '126D': '163',
  '127A': '164',
  '127B': '160',
  '127C': '160',
  '127D': '163',
  '128A': '140',
  '128B': '159',
  '128C': '163',
  '128D': '148',
  '129A': '140',
  '129B': '140',
  '129C': '144',
  '129D': '143',
  '130A': '140',
  '130B': '144',
  '130C': '143',
  '130D': '148',
  '131A': '165',
  '131B': '146',
  '131C': '145',
  '131D': '141',
  '132A': '146',
  '132B': '141',
  '132C': '141',
  '132D': '144',
  '133A': '165',
  '133B': '164',
  '133C': '163',
  '133D': '144',
  '134A': '129',
  '134B': '131',
  '134C': '130',
  '134D': '135',
  '135A': '146',
  '135B': '145',
  '135C': '131',
  '135D': '135',
  '136A': '146',
  '136B': '137',
  '136C': '136',
  '136D': '135',
  '137A': '125',
  '137B': '125',
  '137C': '120',
  '137D': '136',
  '138A': '136',
  '138B': '120',
  '138C': '120',
  '138D': '119',
  '139A': '137',
  '139B': '136',
  '139C': '120',
  '139D': '120',
  '140A': '129',
  '140B': '130',
  '140C': '130',
  '140D': '135',
  '141A': '132',
  '141B': '132',
  '141C': '131',
  '141D': '117',
  '142A': '117',
  '142B': '116',
  '142C': '119',
  '142D': '119',
  '143A': '117',
  '143B': '117',
  '143C': '117',
  '143D': '120',
  '144A': '117',
  '144B': '117',
  '144C': '117',
  '144D': '119',
  '145A': '117',
  '145B': '119',
  '145C': '119',
  '145D': '119',
  '146A': '125',
  '146B': '120',
  '146C': '120',
  '146D': '120',
  '147A': '125',
  '147B': '120',
  '147C': '120',
  '147D': '120',
  '148A': '120',
  '148B': '120',
  '148C': '122',
  '148D': '122',
  '149A': '116',
  '149B': '116',
  '149C': '116',
  '149D': '121',
  '150A': '116',
  '150B': '116',
  '150C': '116',
  '150D': '119',
  '151A': '99',
  '151B': '99',
  '151C': '99',
  '151D': '98',
  '152A': '106',
  '152B': '106',
  '152C': '103',
  '152D': '103',
  '153A': '100',
  '153B': '99',
  '153C': '99',
  '153D': '84',
  '154A': '115',
  '154B': '116',
  '154C': '116',
  '154D': '119',
  '155A': '121',
  '155B': '92',
  '155C': '153',
  '155D': '92',
  '156A': '93',
  '156B': '93',
  '156C': '93',
  '156D': '92',
  '157A': '121',
  '157B': '121',
  '157C': '121',
  '157D': '153',
  '158A': '89',
  '158B': '89',
  '158C': '92',
  '158D': '92',
  '159A': '28',
  '159B': '73',
  '159C': '73',
  '159D': '31',
  '160A': '87',
  '160B': '86',
  '160C': '104',
  '160D': '89',
  '161A': '87',
  '161B': '87',
  '161C': '89',
  '161D': '89',
  '162A': '87',
  '162B': '87',
  '162C': '86',
  '162D': '89',
  '163A': '69',
  '163B': '68',
  '163C': '87',
  '163D': '86',
  '164A': '54',
  '164B': '71',
  '164C': '71',
  '164D': '89',
  '165A': '58',
  '165B': '54',
  '165C': '71',
  '165D': '73',
  '166A': '61',
  '166B': '43',
  '166C': '54',
  '166D': '53',
  '167A': '53',
  '167B': '53',
  '167C': '53',
  '167D': '71',
  '168A': '50',
  '168B': '50',
  '168C': '53',
  '168D': '71',
  '169A': '35',
  '169B': '50',
  '169C': '50',
  '169D': '50',
  '170A': '50',
  '170B': '53',
  '170C': '53',
  '170D': '53',
  '171A': '37',
  '171B': '54',
  '171C': '53',
  '171D': '53',
  '172A': '55',
  '172B': '38',
  '172C': '54',
  '172D': '53',
  '173A': '38',
  '173B': '37',
  '173C': '53',
  '173D': '29',
  '174A': '43',
  '174B': '39',
  '174C': '39',
  '174D': '39',
  '175A': '43',
  '175B': '38',
  '175C': '38',
  '175D': '37',
  '176A': '19',
  '176B': '43',
  '176C': '39',
  '176D': '39',
  '177A': '43',
  '177B': '42',
  '177C': '39',
  '179D': '29',
  '177D': '39',
  '178A': '19',
  '178B': '38',
  '178C': '37',
  '178D': '37',
  '179A': '15',
  '179B': '15',
  '179C': '37',
  '180A': '15',
  '180B': '15',
  '180C': '15',
  '180D': '3',
  '181A': '15',
  '181B': '15',
  '181C': '15',
  '181D': '30',
  '182A': '15',
  '182B': '19',
  '182C': '6',
  '182D': '6',
  '183A': '16',
  '183B': '16',
  '183C': '15',
  '183D': '15',
  '184A': '19',
  '184B': '15',
  '184C': '258',
  '184D': '258',
  '185A': '13',
  '185B': '15',
  '185C': '258',
  '185D': '3',
  '186A': '258',
  '186B': '258',
  '186C': '251',
  '186D': '250',
  '187A': '16',
  '187B': '16',
  '187C': '16',
  '187D': '256',
  '188A': '155',
  '188B': '154',
  '188C': '154',
  '188D': '264',
  '189A': '137',
  '189B': '149',
  '189C': '149',
  '189D': '148',
  '190A': '149',
  '190B': '149',
  '190C': '154',
  '190D': '154',
  '191A': '122',
  '191B': '122',
  '191C': '264',
  '191D': '154',
  '192A': '121',
  '192B': '148',
  '192C': '148',
  '192D': '153',
  '193A': '121',
  '193B': '121',
  '193C': '121',
  '193D': '121',
  '194A': '122',
  '194B': '122',
  '194C': '93',
  '194D': '121',
  '195A': '109',
  '195B': '122',
  '195C': '121',
  '195D': '121',
  '196A': '122',
  '196B': '121',
  '196C': '121',
  '196D': '92',
  '197A': '112',
  '197B': '112',
  '197C': '109',
  '197D': '122',
  '198A': '122',
  '198B': '154',
  '198C': '93',
  '198D': '93',
  '199A': '95',
  '199B': '94',
  '199C': '76',
  '199D': '91',
  '200A': '47',
  '200B': '46',
  '200C': '58',
  '200D': '58',
  '201A': '63',
  '201B': '63',
  '201C': '22',
  '201D': '10',
  '202A': '229',
  '202B': '265',
  '202C': '155',
  '202D': '154'
}

# Define UCL color names
UCL_NAME = {
  '2': 'Strong Pink',
  '3': 'Deep Pink',
  '4': 'Light Pink',
  '5': 'Moderate Pink',
  '6': 'Dark Pink',
  '7': 'Pale Pink',
  '10': 'Pale Gray',
  '11': 'Vivid Red',
  '12': 'Strong Red',
  '13': 'Deep Red',
  '15': 'Moderate Red',
  '16': 'Dark Red',
  '19': 'Grayish Red',
  '22': 'rd Gray',
  '25': 'Vivid Yellowish pink',
  '26': 'Strong Yellowish Pink',
  '27': 'Deep Yellowish Pink',
  '28': 'Light Yellowish Pink',
  '29': 'Moderate Yellowish Pink',
  '30': 'Dark Yellowish Pink',
  '31': 'Pale Yellowish Pink',
  '34': 'Vivid Reddish Orange',
  '35': 'Strong Reddish Orange',
  '37': 'Moderate Reddish Orange',
  '38': 'Dark Reddish Orange',
  '39': 'Grayish Reddish Orange',
  '42': 'Light Reddish Brown',
  '43': 'Moderate Reddish Brown',
  '46': 'Grayish Reddish Brown',
  '47': 'Dark Grayish Reddish Brown',
  '48': 'Vivid Orange',
  '49': 'Brilliant Orange',
  '50': 'Strong Orange',
  '52': 'Light Orange',
  '53': 'Moderate Orange',
  '54': 'Brownish Orange',
  '55': 'Strong Brown',
  '58': 'Moderate Brown',
  '61': 'Grayish Brown',
  '63': 'Light Gray',
  '66': 'Vivid Orangish Yellow',
  '67': 'Brilliant Orangish Yellow',
  '68': 'Strong Orangish Yellow',
  '69': 'Deep Orangish Yellow',
  '70': 'Light Orangish Yellow',
  '71': 'Moderate Orangish Yellow',
  '73': 'Pale Orangish Yellow',
  '76': 'Light Yellowish Brown',
  '82': 'Vivid Yellow',
  '83': 'Brilliant Yellow',
  '84': 'Strong Yellow',
  '86': 'Light Yellow',
  '87': 'Moderate Yellow',
  '89': 'Pale Yellow',
  '91': 'Dark Grayish Yellow',
  '92': 'Yellowish white',
  '93': 'Yellowish Gray',
  '94': 'Light Olive Brown',
  '95': 'Moderate Olive Brown',
  '97': 'Vivid Greenish Yellow',
  '98': 'Brilliant Greenish Yellow',
  '99': 'Strong Greenish Yellow',
  '100': 'Deep Greenish Yellow',
  '101': 'Light Greenish Yellow',
  '103': 'Dark Greenish Yellow',
  '104': 'Pale Greenish Yellow',
  '106': 'Light Olive',
  '109': 'Light Grayish Olive',
  '112': 'Light Olive Gray',
  '115': 'Vivid Yellowish Green',
  '116': 'Brilliant Yellowish Green',
  '117': 'Strong Yellowish Green',
  '119': 'Light Yellowish Green',
  '120': 'Moderate Yellowish Green',
  '121': 'Pale Yellowish Green',
  '122': 'Grayish Yellowish Green',
  '125': 'Moderate Olive Green',
  '129': 'Vivid Yellowish Green',
  '130': 'Brilliant Yellowish Green',
  '131': 'Strong Yellowish Green',
  '132': 'Deep Yellowish Green',
  '135': 'Light Yellowish Green',
  '136': 'Moderate Yellowish Green',
  '137': 'Dark Yellowish Green',
  '140': 'Brilliant Green',
  '141': 'Strong Green',
  '143': 'Very Light Green',
  '144': 'Light Green',
  '145': 'Moderate Green',
  '146': 'Dark Green',
  '148': 'Very Pale Green',
  '149': 'Pale Green',
  '153': 'Greenish white',
  '154': 'Light Greenish Gray',
  '155': 'Greenish Gray',
  '159': 'Brilliant Bluish Green',
  '160': 'Strong Bluish Green',
  '162': 'Very Light Bluish Green',
  '163': 'Light Bluish Green',
  '164': 'Moderate Bluish Green',
  '165': 'Dark Bluish Green',
  '168': 'Brilliant Greenish Blue',
  '169': 'Strong Greenish Blue',
  '171': 'Very Light Greenish Blue',
  '172': 'Light Greenish Blue',
  '173': 'Moderate Greenish Blue',
  '174': 'Dark Greenish Blue',
  '176': 'Vivid Blue',
  '177': 'Brilliant Blue',
  '178': 'Strong Blue',
  '179': 'Deep Blue',
  '180': 'Very Light Blue',
  '181': 'Light Blue',
  '182': 'Moderate Blue',
  '183': 'Dark Blue',
  '184': 'Very Pale Blue',
  '185': 'Pale Blue',
  '189': 'Bluish white',
  '194': 'Vivid Purplish Blue',
  '195': 'Brilliant Purplish Blue',
  '196': 'Strong Purplish Blue',
  '197': 'Deep Purplish Blue',
  '198': 'Very Light Purplish Blue',
  '199': 'Light Purplish Blue',
  '202': 'Very Pale Purplish Blue',
  '204': 'Grayish Purplish Blue',
  '205': 'Vivid Violet',
  '206': 'Brilliant Violet',
  '207': 'Strong Violet',
  '208': 'Deep Violet',
  '210': 'Light Violet',
  '211': 'Moderate Violet',
  '213': 'Very Pale Violet',
  '216': 'Vivid Purple',
  '218': 'Strong Purple',
  '219': 'Deep Purple',
  '221': 'Very Light Purple',
  '222': 'Light Purple',
  '223': 'Moderate Purple',
  '224': 'Dark Purple',
  '226': 'Very Pale Purple',
  '229': 'Dark Grayish Purple',
  '236': 'Vivid Reddish Purple',
  '237': 'Strong Reddish Purple',
  '238': 'Deep Reddish Purple',
  '240': 'Light Reddish Purple',
  '247': 'Strong Purplish Pink',
  '248': 'Deep Purplish Pink',
  '249': 'Light Purplish Pink',
  '250': 'Moderate Purplish Pink',
  '251': 'Dark Purplish Pink',
  '252': 'Pale Purplish Pink',
  '254': 'Vivid Purplish Red',
  '255': 'Strong Purplish Red',
  '256': 'Deep Purplish Red',
  '258': 'Moderate Purplish Red',
  '264': 'Light Gray',
  '265': 'Medium Gray'
}


# Helper function to convert hex to RGB
def hex_to_rgb(hex_color):
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# Convert RGB to CIE-Lab
def rgb_to_lab(rgb_color):
    # Normalizing the RGB values to 0-1 scale for skimage
    rgb_normalized = np.array(rgb_color) / 255.0
    # Converting to LAB using skimage
    return color.rgb2lab(rgb_normalized.reshape(1, 1, 3)).flatten()
    

# Calculate the CIE-Lab distance
def cie_lab_distance(lab1, lab2):
    return np.linalg.norm(lab1 - lab2)

# Find the top 5 closest colors
def find_closest_colors(target_hex, n=5):
    target_rgb = hex_to_rgb(target_hex)
    target_lab = rgb_to_lab(target_rgb)

    distances = []
    for name, hex_color in RGB.items():
        rgb = hex_to_rgb(hex_color)
        lab = rgb_to_lab(rgb)
        distance = cie_lab_distance(target_lab, lab)
        distances.append((name, hex_color, distance))

    # Sort by distance and return the top n closest colors
    closest_colors = sorted(distances, key=lambda x: x[2])[:n]
    return closest_colors


def find_closest_colors_in_rgb(target_hex, n=5):
    target_rgb = hex_to_rgb(target_hex)
    target_lab = rgb_to_lab(target_rgb)

    distances = []
    for name, hex_color in RGB.items():
        rgb = hex_to_rgb(hex_color)
        lab = rgb_to_lab(rgb)
        distance = cie_lab_distance(target_lab, lab)
        distances.append((name, rgb, distance))

    # Sort by distance and return the top n closest colors
    closest_colors = sorted(distances, key=lambda x: x[2])[:n]
    return closest_colors

# Example usage: finding closest colors to a sample color 'f8da21' (hex)
# sample_hex = '753f7c'
# top_5_closest = find_closest_colors(sample_hex)
# top_5_closest

def find_closest_colors_with_ucl(target_hex, n=5):
    target_rgb = hex_to_rgb(target_hex)
    target_lab = rgb_to_lab(target_rgb)

    distances = []
    for name, hex_color in RGB.items():
        rgb = hex_to_rgb(hex_color)
        lab = rgb_to_lab(rgb)
        distance = cie_lab_distance(target_lab, lab)
        ucl = UCL.get(name, "NotFound")  # Fetch UCL value or use "Unknown" if not found
        ucl_name = UCL_NAME.get(ucl, "Unknown")  # Fetch UCL name or use "Unknown" if not found
        distances.append((name, rgb, distance, ucl_name))

    # Sort by distance and return the top n closest colors
    closest_colors = sorted(distances, key=lambda x: x[2])[:n]
    return closest_colors

# Example usage: finding closest colors to a sample color 'f8da21' (hex) with UCL names
# top_5_closest_with_ucl = find_closest_colors_with_ucl(sample_hex)
# top_5_closest_with_ucl

def rgb_to_hex(r, g, b):
    """Convert RGB values to hexadecimal string."""
    return f"{r:02x}{g:02x}{b:02x}"

def get_color_info(rgb_hex):
    """Retrieve UCL and UCL name based on the given RGB hex value."""
    # Find the key in RGB where the value matches the provided rgb_hex
    matching_codes = [code for code, color in RGB.items() if color == rgb_hex]
    if not matching_codes:
        print("RGB value not found.")
        return None, None, None

    # Retrieve UCL and UCL name for the first matching code
    code = matching_codes[0]
    ucl_value = UCL.get(code)
    ucl_name = UCL_NAME.get(ucl_value, "Unknown")

    return code, ucl_value, ucl_name


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Retrieve color mappings.")
    parser.add_argument("--rgb", type=str, help="RGB value to look up in the format 'R,G,B' (e.g., '229,227,161')")
    args = parser.parse_args()

    if args.rgb:
        # Convert RGB input to hexadecimal
        try:
            r, g, b = map(int, args.rgb.split(","))
            rgb_hex = rgb_to_hex(r, g, b)
            print(f"Converted RGB to hex: {rgb_hex}")

            top_5_closest_in_rgb = find_closest_colors_with_ucl(rgb_hex)
            # Display the results
            print("Top 5 closest colors:")
            for color in top_5_closest_in_rgb:
                label, rgb, distance, ucl_name = color
                print(f"Label: {label}, RGB: {rgb}, Distance: {distance:.2f}, UCL Name: {ucl_name}")
        except ValueError:
            print("Please provide the RGB value in the correct format 'R,G,B'.")
    else:
        print("Please specify an RGB value using --rgb.")

if __name__ == "__main__":
    main()
