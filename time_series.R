

### week and month to the data 
library(lubridate)
library(dplyr)
library(ggplot2)
library(readr)
retail_clean<- read_csv('retail_clean.csv')
retail_clean$week<- week(retail_clean$InvoiceDate)
retail_clean$revenue<- retail_clean$Price *retail_clean$Quantity
retail_clean$year<- year(retail_clean$InvoiceDate)

time_series<-retail_clean %>% group_by(week,year) %>%
  summarise(date=mean(InvoiceDate),
             revenue=sum(revenue,na.rm = TRUE)) %>% arrange(date)

time_series_sales<-retail_clean %>% group_by(week,year) %>%
  summarise(date=mean(InvoiceDate),
            sales=sum(Quantity,na.rm = TRUE)) %>% arrange(date)


time_series %>% ggplot(aes(x=date,y=revenue))+geom_line()

time_series$date<- as.Date(time_series$date)

write_csv(time_series,"time_series_revenue.csv")
write_csv(time_series_sales,"time_series_revenue.csv")


#### Linear regression Model

time_series

time_series$trend<- seq(1: nrow(time_series))
time_series$month<-month(time_series$date,label = TRUE)
time_series$dayofmonth<- day(time_series$date)
time_series<-time_series %>%
  mutate(first_10= case_when(dayofmonth <= 10 ~ 1,
                             dayofmonth >10 ~ 0))

time_series<-time_series %>%
  mutate(secoond_10= ifelse(dayofmonth >10 & dayofmonth <=20,1,0))
                            
time_series<-time_series %>%
  mutate(third_10= ifelse(dayofmonth > 20,1,0))
glimpse(time_series)

colnames(time_series)

moodel_1<- lm(revenue ~ .,data = time_series[,c(2,5,6,8,9,10)])

summary(moodel_1)


time_series$prediction<- predict(moodel_1,time_series)


mae<- mean(abs(time_series$revenue-time_series$prediction))



####Making the forecast
future_dates<- seq.Date(from = max(time_series$date)+7,
                        to= max(time_series$date)+ (7*16),by=7)

future_dates<- data.frame(date=future_dates)

time_series$type<- "History"

nrow(time_series)

time_series<- rbind(time_series,future_dates)

time_series$type[is.na(time_series$type)==TRUE]<- "Forecast"

table(time_series$type)


###redoing it again
time_series$trend<- seq(1: nrow(time_series))
time_series$month<-month(time_series$date,label = TRUE)
time_series$dayofmonth<- day(time_series$date)
time_series<-time_series %>%
  mutate(first_10= case_when(dayofmonth <= 10 ~ 1,
                             dayofmonth >10 ~ 0))

time_series<-time_series %>%
  mutate(secoond_10= ifelse(dayofmonth >10 & dayofmonth <=20,1,0))

time_series<-time_series %>%
  mutate(third_10= ifelse(dayofmonth > 20,1,0))
glimpse(time_series)


time_series$prediction<-predict(moodel_1,time_series)


a<-time_series %>% ggplot(aes(x=date,y=prediction,color=type))+geom_line()+
  geom_line(aes(y=revenue),color="darkblue")+theme_minimal()

library(plotly)


ggplotly(a)


































