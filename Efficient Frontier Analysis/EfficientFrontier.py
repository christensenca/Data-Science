#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 06:54:27 2020

@author: cadechristensen
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import quandl
import random
import os
import enum
import statistics

####JSON file location
stock_1 = '/Users/cadechristensen/Downloads/FSKAX.json'
stock_2 = '/Users/cadechristensen/Downloads/FSPSX.json'
stock_3 = '/Users/cadechristensen/Downloads/FGOVX.json'
stock_files = [stock_1, stock_2, stock_3]
#######################################################
    
##################JSON Stocks###########################
ticker_1 = 'FSKAX'
ticker_2 = 'FSPSX'
ticker_3 = 'FGOVX'
#######################################################

##################API Information#######################
quandl.ApiConfig.api_key = "osfQD5dZCxECdQkoqVsV"
stock_list = ['AAPL', 'GOOG','FB', 'AMZN']
number_of_data_points = 10000
    
class CreateDataSetAPI:
    """This Class Reads in the data from quandl's API and Formats it as a dataframe, 
and creates the Weighted Variables"""
    def __init__(self):
        """loads the names of the stocks in the portfolio and properly formats them into a list.
        checks the inputs to ensure valid ticker symbols"""
        stock_list = []
        raw_input = input('Enter Stock Tickers separated by a comma and in all caps (ie:FB, GOOG, AMZN, AAPL):')
        x = raw_input.split(',')
        for i in x:
            stock = i.replace(' ', '')
            stock_list.append(stock)
        for i in stock_list:
            if len(i) > 5:
                raise ValueError('{} is invalid. Valid ticker symbols are between two and five characters'.format(i))
               
            if len(i) < 2:
                raise ValueError('{} is invalid. Valid ticker symbols are between two and five characters'.format(i))        
        self.stock_list = stock_list
    
    
        
    def load_stocks_from_api(self):
        """Checks to make sure the start and end dates are properly formatted
        This function takes a list of stocks and reads them into a dataframe from quandl
   sets the index on the date, and pivots the table to the ticker column . 
   the stocks are read in twice, one list captures
   the open price and the second captures the close price. The to lists are captured
    into a dictionary with keys called 'Open' and 'Close'"""
        
        start_date = input('Enter start date formatted as YYYY-MM-DD (ie: 2016-10-10):')
        end_date = input('Enter end date formatted as YYYY-MM-DD (ie 2017-10-10):')
        if (len(start_date) or len(end_date)) != 10:
            raise ValueError('Your start date or end date was not formatted correctly. Try again')
        elif start_date == end_date:
            raise ValueError('Your start and end date are the same. Make sure your start and end dates are different')
        elif start_date > end_date:
            raise ValueError('Your end date is before your start date.')
        open_list = []
        close_list = []
        PriceData = {}
        for TickerName in self.stock_list:
            data = quandl.get_table('WIKI/PRICES', qopts = { 'columns': ['ticker', 'date', 'open'] }, ticker = TickerName, date = { 'gte':start_date, 'lte': end_date }, paginate = True)
            df = data.set_index('date')
            table = df.pivot(columns='ticker')
            table.columns = [col[1] for col in table.columns]
            open_list.append(table)
        for TickerName in self.stock_list:
            data = quandl.get_table('WIKI/PRICES', qopts = { 'columns': ['ticker', 'date', 'close'] }, ticker = TickerName, date = { 'gte': start_date, 'lte': end_date }, paginate = True)
            df = data.set_index('date')
            table = df.pivot(columns='ticker')
            table.columns = [col[1] for col in table.columns] 
            close_list.append(table)
        open_df = pd.concat(open_list, axis = 1)
        close_df = pd.concat(close_list, axis = 1)
        PriceData['Open'] = open_df
        PriceData['Close'] = close_df
        return PriceData
    

    def PortfolioWeightsAPI(self, number_of_data_points:int):
         """ portfolio weights assigns a weight to each stock in the portfolio
   and saves it as a dataframe. ie, 25%, 25%, 50%"""
         column_names = [] 
         df = []
         t = 0
         while t < number_of_data_points:
             weights = [random.random() for _ in range(len(self.stock_list))]
             sum_weights = sum(weights)
             weights = [w/sum_weights for w in weights]
             df.append(weights)
             t = t + 1
         make_df = pd.DataFrame(df) 
         for i in self.stock_list:
             column_names.append(i + ' Weight')
         make_df.columns = column_names
         return make_df

class CreateDataSetsJSON:
    """The initializer for this class takes the file paths of the json files as a list 
    and the tickers associated with those files"""
    def  __init__(self, ticker_1, ticker_2, ticker_3, stock_files, start_date, end_date):
        self.ticker_1 = ticker_1
        self.ticker_2 = ticker_2
        self.ticker_3 = ticker_3
        self.stock_files = stock_files
        self.start_date = start_date
        self.end_date = end_date
         

    def load_json_stock_file(self):
        """This function iteratively reads in each json stock file; selects 
      the date, open, and close price and sets the index on the date column.
      saving the dataframe to a list of dataframes. Then the the individual dataframes
      are concatenated and saved to a dictionary by open and close price.
      The open and close prices are assigned their tickers"""
        open_list = []
        close_list = []
        PriceData = {}
        for stock in stock_files:
            with open(stock) as file:
                df_stock = pd.read_json(file)
                select_column = df_stock[['Date','Open', 'Close']]
                select_column.columns = ['date', 'Open', 'Close']
                set_index = select_column.set_index('date')
                select_date_range = set_index.loc[self.start_date:self.end_date]
                close_list.append(select_date_range['Close'])
                open_list.append(select_date_range['Open'])
                compile_open_sets = pd.concat(open_list, axis = 1)
        compile_close_sets = pd.concat(close_list, axis = 1)
        compile_open_sets.columns = [self.ticker_1, self.ticker_2, self.ticker_3]
        compile_close_sets.columns = [self.ticker_1, self.ticker_2, self.ticker_3]
        PriceData["Open"] = compile_open_sets
        PriceData['Close'] = compile_close_sets
        return PriceData
  


    def PortfolioWeights(self, number_of_data_points):
        """The PortfolioWeights function creates a random portfolio 
   allocation and assigns it to a stock. if for example, 
   if there are four stocks in a portfolio, then an allocation will:
   be created (.25, .25, .25, .25) that pertains to a stock in the portfolio"""
        t = 0
        x = []
        y = []
        z = []
    
        weight_dict = {}
        while t < number_of_data_points:
            n = 3
            weights = [random.random() for _ in range(n)]
            sum_weights = sum(weights)
            weights = [w/sum_weights for w in weights]
            x.append(round(weights[0], 2))
            y.append(round(weights[1], 2))
            z.append(round(weights[2], 2))
            t = t + 1
        weight_dict[self.ticker_1] = x
        weight_dict[self.ticker_2] = y
        weight_dict[self.ticker_3] = z
        portfolio_weight = pd.DataFrame.from_dict(weight_dict)
        return portfolio_weight


class PortfolioStatistics:
    """This Class calculates the basic measures of stock performance 
    Percent Yield and Standard Deviation"""
    def __init__(self, price_data):   
        self.price_data = price_data
        
    
    
    def calculate_years(self)->int:
        """This function converts the time between the first entry
    and the last entry of the index to  a float representing years"""
        ReIndex = self.price_data.reset_index()
        date = ReIndex['date']
        time = (date.iloc[-1] - date.iloc[0])
        days = time.days
        return days/365       
     

    def percent_return_dict(self)->dict:
        """This function calculates each stocks yearly percent return
    and captures it in a dictionary"""
        returns_yearly = (((self.price_data.iloc[-1] - self.price_data.iloc[0])/self.price_data.iloc[-1])/self.calculate_years()) * 100
        return returns_yearly.to_dict()
    
    
   
    def standard_deviation_dict(self)-> dict:
        """This function calculates the yearly standard deviation 
     and captures it in a dictionary"""
        df = self.price_data
        stock_std = df.std()/self.calculate_years()
        return stock_std.to_dict()

class MakeCoordinates:
     """This Class creates the Efficient Frontier Coordinates for the Graph by making a 
     composite function of return and its respective weight
     and the Standard Deviation and its Respective weight. 
     IE; y = weight_1(stock_1's return) + weight_2(stock_2's return) +.... ; 
     x = weight_1(stock_1's stdev) + weight_2(stock_2's stdev)"""
     def __init__(self, portfolio_weights, pct_return, stdev):
         self.portfolio_weights = portfolio_weights
         self.pct_return = pct_return
         self.stdev = stdev

       
     def return_vs_std_coordinates(self):
         """This function creates the coordinates for the Percent Return and the 
    Standard Deviation with respect to its weight. 
    IE; y = weight_1(stock_1's return) + weight_2(stock_2's return) +.... ;
    x = weight_1(stock_1's stdev) + weight_2(stock_2's stdev)"""
         raw_percents = pd.DataFrame()  
         raw_stdev = pd.DataFrame()
         ColumnNames = self.pct_return.keys()       
         for column in ColumnNames:
             raw_percents[column] = self.portfolio_weights[column + ' Weight'] * self.pct_return[column]
             raw_stdev[column] = self.portfolio_weights[column + ' Weight'] * self.stdev[column]
         raw_percents['TotalPct'] = raw_percents.sum(axis = 1)
         raw_stdev['TotalStdev'] = raw_stdev.sum(axis = 1)
         self.portfolio_weights['PercentReturn'] = raw_percents['TotalPct']
         self.portfolio_weights['StandardDeviation'] = raw_stdev['TotalStdev']
         return self.portfolio_weights


class MakeGraphs:
    """This Class creates the graphs and tables for the Efficient Frontier analysis. """
    def __init__(self, PriceData):
        self.PriceData = PriceData
    
    def table_efficient_frontier(self, portfolio_weights):
        """takes the points used to make the efficient frontier
        and finds the max return and the minimum volatility to make a
        table"""
        price_data = self.PriceData['Open']
        portfolio_statistics = PortfolioStatistics(price_data)
        pct_return = portfolio_statistics.percent_return_dict()
        stdev = portfolio_statistics.standard_deviation_dict()
        make_coordinates = MakeCoordinates(portfolio_weights, pct_return, stdev)
        EFPoints = make_coordinates.return_vs_std_coordinates()
        MinVol = EFPoints[EFPoints['StandardDeviation'] == EFPoints['StandardDeviation'].min()]
        MinVol['Portfolio Optimization'] = ['Minimize Volatility']
        MaxReturn = EFPoints[EFPoints['PercentReturn'] == EFPoints['PercentReturn'].max()]
        MaxReturn['Portfolio Optimization'] = ['Maximize Return']
        EFTable  = pd.concat([MinVol, MaxReturn])
        df = EFTable.set_index('Portfolio Optimization')
        final_df = df.reset_index()
        return final_df
        
         
   
#######################Generates the report sections#############################
class ReportPartType(enum.Enum):
    """Identifies part of report"""
    Paragraph = 0 
    Section = 1
    Figure = 2
    Table = 3


class ReportPart:
    """initializes part_number at 0 
    for generation of html"""
    def __init__(self):
        self.part_number :int = 0
        self.part_type_number: int = 0
    pass

class CrossReference:
        """creates a crossreference to a 
        part of the report"""
        def __init__(self, report_part):
            self.report_part = report_part
            
            
        def __str__(self)->str:
            """overloads the str function to add a crossreference"""
            return '{} {}'.format(type(self.report_part),self.report_part.part_type_number)

class Paragraph(ReportPart):
    """Adds a paragraph or crossreference to a section of the report"""
    def __init__(self, text_parts=None):
        if text_parts is not None:
            self.text_parts = text_parts
        else:
            self.text_parts = []
        
    def append(self, text:str):
        """appends paragraph"""
        self.text_parts.append(text)


    def append_cross_reference(self, report_part):
        """appends cross reference"""
        self.text_parts.append(CrossReference(report_part))
        
    def get_type(self):
        return ReportPartType.Paragraph
    
    
class Figure(ReportPart):
    """creates a figure for the report
    and saves the figure"""
    def __init__(self):
        super().__init__()
        self.matplotlib_figure = plt.figure(figsize=(15,15))
        self.caption = ''
    
    def save_to_file(self, file_path:str):
        """saves plot from graph functions"""
        self.matplotlib_figure.savefig(file_path, bbox_inches='tight')
        
    def get_type(self):
        return ReportPartType.Figure     
    
    def graph_open_close_prices(self, PriceData):
        """daily open and close graphs, used for traded stocks and 
        formats subplots according to the number of stocks"""
        Open =  PriceData['Open']
        Close = PriceData['Close']
        rowCnt = len(Open.columns)
        colCnt = 2
        subCnt = 1
        fig = self.matplotlib_figure
        self.caption = 'Opening and Closing Prices Over Time'
        for c in Open.columns:
            fig.add_subplot(rowCnt, colCnt, subCnt)
            plt.plot(Open.index, Open[c], lw = 2, alpha = .80, color = 'green')
            plt.plot(Close.index, Close[c], lw = 2, alpha = .80, color = 'red')
            plt.ylabel('Prices in US Dollars')
            red = mpatches.Patch(color='red', label='Close')
            green = mpatches.Patch(color = 'green', label= 'Open')
            plt.legend(loc = 'upper left',  handles=[red, green])            
            plt.title(c + ' Historical Prices')
            subCnt = subCnt + 1
         
    
    def chart_stdev_and_pct_return(self, PriceData):
        """annual standard deviation and perecent return bar chart. Percent
        return is a bar chart on one axis and the standard deviation is line on the 
        second axis"""  
        price_data = PriceData['Open']
        portfolio_statistics = PortfolioStatistics(price_data)
        stdev = portfolio_statistics.standard_deviation_dict()
        stdev_df = pd.DataFrame([stdev],columns = stdev.keys())
        pivot_stdev_df = stdev_df.iloc[0] 
        pct_return = portfolio_statistics.percent_return_dict()
        return_df = pd.DataFrame([pct_return], columns = pct_return.keys())
        pivot_return_df = return_df.iloc[0]
        self.caption = 'Standard Deviation and Percent Return'
        self.matplotlib_figure, ax1 = plt.subplots()
        self.matplotlib_figure.set_size_inches(8,8)
        ax2 = ax1.twinx()
        ax1.plot(pivot_stdev_df,  color = 'blue')
        ax2.bar(pivot_return_df.index, pivot_return_df ,alpha = .2, color = 'green')
        ax1.set_ylabel('Standard Deviation', color = 'blue')
        ax2.set_ylabel('Percent Return', color = 'green')
        plt.title('Annual Returns vs Standard Deviation')
        
    def graph_efficient_frontier(self, PriceData, portfolio_weights):
        """takes the points used to make the efficient frontier
        and finds the max return and the minimum volatility to make a
        table"""
        price_data = PriceData['Open']
        portfolio_statistics = PortfolioStatistics(price_data)
        pct_return = portfolio_statistics.percent_return_dict()
        stdev = portfolio_statistics.standard_deviation_dict()
        make_coordinates = MakeCoordinates(portfolio_weights, pct_return, stdev)
        EFPoints = make_coordinates.return_vs_std_coordinates()
        self.caption = 'Efficient Frontier Analysis'
        self.matplotlib_figure.set_size_inches(8,8)
        plt.scatter(x = EFPoints['StandardDeviation'], y = EFPoints['PercentReturn'])
        plt.ylabel('Annual Percent Return')
        plt.xlabel('Annual Standard Deviation')
        plt.title('Efficient Frontier Analysis')
        
    def graph_daily_prices(self, PriceData):
        """graphs only the closing prices, used for index funds"""
        price_data = PriceData['Open']
        self.matplotlib_figure   
        for c in price_data.columns.values:
            plt.plot(price_data.index, price_data[c], lw=3, alpha=0.8,label=c)
            plt.legend(loc='upper left')
            plt.ylabel('Price in US Dollars')
            plt.xlabel('Date')
            plt.title('Price Over Time')
    
    def graph_daily_percent_return(self, PriceData):
        """creates a graph of daily percent return for 
        each stock in the portfolio and formats the 
        subplots according to the number of stocks """
        price_data = PriceData['Close']
        daily_returns = price_data.pct_change()
        rowCnt = len(daily_returns.columns)
        colCnt = 2
        subCnt = 1
        self.caption = 'Daily Percent Return'
        fig = self.matplotlib_figure
        fig.set_size_inches(14,16)
        for c in daily_returns.columns:
            fig.add_subplot(rowCnt, colCnt, subCnt)
            plt.plot(price_data.index, daily_returns[c], lw = 3, alpha = 0.80, label = c)
            plt.ylabel('Daily Percent Return')
            plt.title(c + ' Daily Returns')
            subCnt = subCnt + 1
        
class Table(ReportPart):
    """creates an html table"""
    def __init__(self):
        super().__init__()
        self.header = []
        self.data = []
        
    def set_header(self, header_names):
        """header for each table"""
        for header_name in header_names:
            self.header.append(header_name)
            
    def set_data(self, data):
        """data rows for table"""
        self.data = data
            
    def get_type(self):
        return ReportPartType.Paragraph


class Section(ReportPart):
    def __init__(self, title:str, children= None):
        """creates a section of a reort and adds paragraphs,
        figures and tables to each section"""
        self.title = title
        if children is not None:
            self.children = children
        else:
            self.children = []
            
    def add_section(self, section_title):
        """adds a new sections"""
        new_section = Section(section_title)
        self.children.append(new_section)
        return new_section
    
    def add_paragraph(self):
        """adds a Paragraph class to a section"""
        new_para = Paragraph()
        self.children.append(new_para)
        return new_para
    
    def add_table(self):
        """adds a table class to a section"""
        new_table = Table()
        self.children.append(new_table)
        return new_table
    
    def add_figure(self):
       """adds a figure to a section"""
       new_figure = Figure()
       self.children.append(new_figure)
       return new_figure

    def get_type(self):
        return ReportPartType.Section
    

class Report:
    def __init__(self, title:str, sections=None):
        """initializes the report
       a report has a series of sections"""
        self.title = title
        if sections is not None:
            self.sections = sections
        else:
            self.sections = []
        
    def add_section(self, section_title):
        """adds a section in a report"""
        new_section = Section(section_title)
        self.sections.append(new_section)
        return new_section
    
class HTMLReportContext:
    """generates the html file from the report sections"""
    def __init__(self, folder_path:str):
        self.folder_path = folder_path
        self.figure_count = 0
    def _generate_section(self, section, html_file, level:int = 2):
        """is called by the generate function to generate the html file for each section.
        adds encodes the paragraphs, figures, and tables"""
        html_file.write("<div class = 'section'><h{}>{}</h{}>".format(level, section.title ,level))
        for report_part in section.children:
            if isinstance(report_part, Paragraph):
                html_file.write("<p>")
                for text_part in report_part.text_parts:
                    html_file.write(str(text_part))
                html_file.write("<p>")
            elif isinstance(report_part, Table):
                table: Table = report_part
                html_file.write("<table class = 'Table table-hover'>")
                html_file.write("<tr>")
                for header_name in table.header:
                    html_file.write("<th scope = 'col'>{}</th>".format(header_name))
                html_file.write("</tr>")
                for element in table.data:
                   html_file.write("<tr scope = 'row'>")
                   for data_element in element:
                       html_file.write("<td>{}</td>".format(data_element))
                   html_file.write("</tr>")
                
                html_file.write("</table>")
            elif isinstance(report_part, Figure):
                figure: figure = report_part
                relative_image_path = "{}.png".format(self.figure_count)
                figure.save_to_file(os.path.join(self.folder_path, relative_image_path))
                html_file.write("<img src = '{}'class= 'center-block'/>".format(relative_image_path))
                html_file.write("<figcaption>Figure{}. {}</figcaption>".format(figure.part_type_number, figure.caption))
                self.figure_count +=  1 
            elif isinstance(report_part, Section):   
                self._generate_section(report_part, html_file, level + 1) 
        html_file.write("</div>")
        
    def assign_part_number(self, report_part, number_counts):
           """Assings a part number to each report part and calls itself for every 
           child in a section"""
           t = report_part.get_type()
           number_counts[t] += 1 
           number_counts["Total"] += 1 
           report_part.part_number = number_counts["Total"]
           report_part.part_type_number = number_counts[t]
           if t == ReportPartType.Section:
               for child in report_part.children:
                   self.assign_part_number(child, number_counts)
           
    def generate(self, report, file_name:str):
        """generates the body of the html file, encodes the style sheet from boostrap, and calls the generate_section
        for each section of the report"""
        number_counts= {ReportPartType.Paragraph:0, ReportPartType.Figure:0, ReportPartType.Table:0, ReportPartType.Section:0, "Total":0}
        for section in report.sections:
            self.assign_part_number(section, number_counts)
        with open(os.path.join(self.folder_path,"{}.html".format(file_name)),'w') as html_file:
            html_file.write('<html>')
            html_file.write('<head>')
            html_file.write('<title >{}</title>'.format(report.title))
            html_file.write('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css" integrity="sha384-BVYiiSIFeK1dGmJRAkycuHAHRg32OmUcww7on3RYdg4Va+PmSTsz/K68vbdEjh4u" crossorigin="anonymous">')
            html_file.write("<body>")
            html_file.write("<div class= 'container p-3 mb-2 bg-info text-white' >")
            html_file.write("<div class= 'jumbotron'>")
            html_file.write('<h1 class = "text-center">{}</h1>'.format(report.title))
            for section in report.sections:
                self._generate_section(section, html_file)
            html_file.write("</div>")
            html_file.write("</div>")
            html_file.write("<body>")   
            html_file.write('<head>')
            html_file.write('<html>')         
             
             
             
        
if __name__ == "__main__":        
#####################################################################
#            Initializes the Data sets to make the report
#####################################################################
    create_data_set_api = CreateDataSetAPI()  
    PriceData = create_data_set_api.load_stocks_from_api()
    close_price = PriceData['Close']
    portfolio_weights = create_data_set_api.PortfolioWeightsAPI(100000)
    portfolio_weights2 = portfolio_weights.copy(deep=True)
    make_graphs = MakeGraphs(PriceData)
    portfolio_statistics = PortfolioStatistics(PriceData['Open'])
    pct_return = portfolio_statistics.percent_return_dict()
    stdev = portfolio_statistics.standard_deviation_dict()
   ##find the start date and end date in the dataset
    start_date_with_time = str(PriceData['Open'].index[0])
    start_date_with_time = start_date_with_time.split(' ')
    start_date = start_date_with_time[0]
    end_date_with_time = str(PriceData['Open'].index[-1])
    end_date_with_time = end_date_with_time.split(' ')
    end_date = end_date_with_time[0]
    ####################################################################
#####################################################################
#            Section 1
    # Price over time 
    # shows the highest priced and lowest prices stock
    # graphs the historical daily prices
#####################################################################
    report  = Report('Efficient Frontier Analysis')
    section = report.add_section('Price Performance Over Time')
    paragraph = section.add_paragraph()
    paragraph_1 = section.add_paragraph()
    paragraph_1_1 = section.add_paragraph()
    figure_1 = section.add_figure()
    
    paragraph.append('The goal of the efficient frontier analysis is to balance a portfolio in such a way that returns will be maximized while eliminating the  portfolio volatility. Figure 1, below shows the historical prices for the stocks you have selected from {} to {}. This will give you an overall sense of the health of the company.'.format(start_date, end_date))
    ##Find most valuable stock in portfolio
    most_recent_stock_price = close_price.iloc[-1]
    most_recent_stock_price_dict = most_recent_stock_price.to_dict()
    highest_price = most_recent_stock_price.max()
    lowest_price = most_recent_stock_price.min()
    for key in most_recent_stock_price_dict.keys():
        
        if most_recent_stock_price_dict[key] == highest_price:
            paragraph_1.append('{} (${}) is the most valuable stock in your portfolio'.format(key, highest_price))
        elif most_recent_stock_price_dict[key] == lowest_price:
            paragraph_1_1.append('{} (${}) is the least valuable stock in your portfolio.'.format(key, lowest_price))
    figure_1.graph_open_close_prices(PriceData)
    
#####################################################################
#            Section 2
    # Graphs daily percent return
    # for stock in portfolio_statistics:
        
#####################################################################
   #initialize section 2
    section_2 = report.add_section('Daily Percent Return')
    paragraph_2 = section_2.add_paragraph()
    figure_2 = section_2.add_figure()
    
    paragraph_2.append('Figure 2, shows the daily returns for the stocks you have selected. Large negative fluctuations in daily percent return may influence your portfolios ability to make steady returns.')
    figure_2.graph_daily_percent_return(PriceData)
######################################################################
#            Section 3
    # Show average standard deviation and percent return in 
    # a bar and line chart. Table is generated based on the 
    # chart data. If the percent return for a stock is below the 
    # average return, prints "these stocks performed below average"
#####################################################################
    #####calculate average standard deviation and percent return
    return_list = []
    stdev_list = []
    for key in pct_return.keys():
        return_list.append(pct_return[key])
    average_return = round(statistics.mean(return_list), 2)
    for key in stdev.keys():
        stdev_list.append(stdev[key])
    average_stdev = round(statistics.mean(stdev_list), 2)
    ###Create standard deviation and percent return data for table
    table_1_data = []
    for key in stdev.keys():
        table_1_data.append([key, round(stdev[key], 2), round(pct_return[key], 2)])
    ####initialize Section 3 sections
    section_3 = report.add_section("Your Portfolio's Yearly Standard Deviation and Percent Return")
    paragraph_3 = section_3.add_paragraph()
    paragraph_3_1 = section_3.add_paragraph()
    figure_3 = section_3.add_figure()
    table_1 = section_3.add_table()
    
    paragraph_3.append('Figure 3 compares the average yearly percent return with the average yearly standard deviation. It can be expected that a stock that has a high rate of return and a low standard deviation will compose the majority of your efficient frontier portfolio.')
    ####create a list of stocks that are performing below average
    paragraph_3_1_state = False
    below_average_return = []
    for key in pct_return.keys():
        if round(pct_return[key],2) < average_return:
            below_average_return.append(key)
            paragraph_3_1_state = True
    ######append section 3 
    if paragraph_3_1_state:
        paragraph_3_1.append('The following stocks in your portfolio performed below the average yearly return ({}%): {}.'.format(average_return, ', '.join(below_average_return)))
    figure_3.chart_stdev_and_pct_return(PriceData)
    table_1.set_header(['Stock','Standard Deviation', 'Percent Return'])
    table_1.set_data(table_1_data)
    #####Make Table for Efficient Frontier
    if len(PriceData['Open'].columns) > 1:  
        EFP_df = make_graphs.table_efficient_frontier(portfolio_weights2)
        df_table = EFP_df.round(2)
        data_list = []
        table_row1 = df_table.iloc[0]
        table_row2 = df_table.iloc[1]
        data_list.append(table_row1)
        data_list.append(table_row2)
 #####################################################################
#            Section 4
        # Graphs the efficient frontier
        # if number of stocks selected is 1 or 2 a message is 
        # generated saying the efficient frontier graph will not 
        # provide useful results. Table is generated based on the portfolio 
        # that minimizes standard deviation and the portfolio that maximizes
        #percent return
#####################################################################
    #initialize section 4 sections
    section_4 = report.add_section("Your Portfolio's Efficient Frontier")
    paragraph_4 = section_4.add_paragraph()
    figure_4 = section_4.add_figure()
    table_2 = section_4.add_table()
    
    if len(PriceData['Open'].columns) == 1 :
        paragraph_4.append('Your Efficient Frontier graph is a single point because you have selected only a single stock to analyze. For best results, analyze three or more stocks')
    elif len(PriceData['Open'].columns) == 2:
        paragraph_4.append('Your Efficient Frontier graph is a straight line because you have selected only two stocks to analyze. For best results, analyze three or more stocks')
        table_2.set_header(df_table.columns)
        table_2.set_data(data_list)
    else: 
        paragraph_4.append('Figure 4, is a graph of your efficient frontier. The point of inflection is the location where your portfolio will minimize its volatility over the interval {} to {}. '.format(start_date, end_date))
        table_2.set_header(df_table.columns)
        table_2.set_data(data_list)
   
    figure_4.graph_efficient_frontier(PriceData, portfolio_weights)
    paragraph_4.append('The table below shows the optimized portfolios for the timer interval selected.')
    
    
#####################################################################
#            Generate HTML
#####################################################################
    html_generator = HTMLReportContext('')
    html_generator.generate(report, 'EfficientFrontierAnalysis')
    
    
